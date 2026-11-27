#!/usr/bin/env python3
"""Contact-GraspNet behind the AnyGrasp server's HTTP API (a stand-in until AnyGrasp is licensed).

It speaks the API of ``serve_anygrasp.py`` (``/plan``, ``/plan_viz``, ``/health``,
``/reset_state``) and returns grasps in AnyGrasp's convention, so
``sample_grasp_pose_anygrasp`` and its client (``skill_library.sample_grasp_anygrasp``) work
unchanged. Run it on AnyGrasp's port (8122). To switch back, stop it and start
``serve_anygrasp.py`` on the same port.

Model: the PyTorch port of Contact-GraspNet (https://github.com/elchun/contact_graspnet_pytorch,
weights included; NVIDIA's non-commercial licence). It runs in its own venv, not ENPIRE's, because
it needs only torch/numpy/scipy/trimesh/cv2 and an older numpy. This file therefore imports
nothing from ENPIRE.

Per request:
1. Build the scene cloud from depth and K. The object segment is ``segmap == segmap_id``,
   i.e. the SAM3 mask the client sends.
2. Run Contact-GraspNet on a local region around the segment, keeping only grasps whose contacts
   lie on the segment (``local_regions`` and ``filter_grasps``). The surrounding scene stays in
   the region's input. ``object_input_mode`` is accepted for API compatibility; Contact-GraspNet
   always uses the local region.
3. Convert each pose to AnyGrasp's convention:
   - Contact-GraspNet pose: x = closing axis, z = approach, origin at the gripper base, with
     the fingertip baseline 0.1034 m along z;
   - AnyGrasp pose: x = approach, y = closing axis, origin between the fingers;
   - so R_any = [z, x, y] and t_any = t + (0.1034 + extra_depth) z.
   On the RoboCasa sink objects the model puts contacts on the camera-facing top surface and
   predicts small openings (9-13 mm), so the pads can close above the object's widest part.
   ``--extra-depth`` moves the pose further along the approach. It is experimental: in the
   2026-11-27 probe (scripts/grasp_probe.py, 4 kitchens), neither 2 nor 4 cm produced a lift.
4. If ``--top-down`` is set (the default, like AnyGrasp's top-down mode), keep grasps whose
   approach is within 60 deg of the camera's viewing direction. If none qualify, keep all.
5. Greedy NMS (2 cm, 30 deg), sort by score and return ``max_grasps``.

Contact-GraspNet's scores (about 0.2 to 0.5) are not on AnyGrasp's scale; only their order is
comparable. The server does no collision check, so ``collision_detection`` is ignored. The
client's cuRobo IK and planning filter the poses.

    PYTHONPATH=<cgn_root>:<cgn_root>/Pointnet_Pointnet2_pytorch PYOPENGL_PLATFORM=egl \\
      <cgn venv>/bin/python serve_contact_graspnet.py --cgn-root <cgn_root> --port 8122
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_contact_graspnet")

app = FastAPI(title="Contact-GraspNet server (AnyGrasp API)")

GRIPPER_DEPTH_M = 0.1034  # Contact-GraspNet: gripper frame origin -> fingertip baseline
_ESTIMATOR: Any | None = None
_CKPT: str | None = None
_FORWARD_PASSES = 1
_TOP_DOWN = True
_TOP_DOWN_COS = 0.5  # cos(60 deg)
_EXTRA_DEPTH_M = 0.0  # push past the predicted contacts along the approach (see --extra-depth)
_LOCK = threading.Lock()  # one GPU inference at a time


class PlanRequest(BaseModel):
    rgb_base64: str
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int = 1
    z_range: list[float] | None = None
    max_grasps: int = 20
    workspace_margin: float = 0.02
    collision_detection: bool = True
    object_input_mode: str = "segmented_object_cloud"


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    widths_base64: str
    n_grasps: int
    best_score: float | None = None


class PlanVizResponse(PlanResponse):
    overlay_jpeg_base64: str
    grasp_thumbnail_jpeg_base64: list[str] | None = None


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


def cgn_to_anygrasp(grasps: np.ndarray, extra_depth_m: float = 0.0) -> np.ndarray:
    """Contact-GraspNet poses (N,4,4) -> AnyGrasp-convention poses, both in the camera frame.

    The origin moves to the fingertip baseline (0.1034 m along the approach), plus
    ``extra_depth_m`` further along the approach.
    """
    out = np.tile(np.eye(4), (len(grasps), 1, 1))
    x, y, z = grasps[:, :3, 0], grasps[:, :3, 1], grasps[:, :3, 2]
    out[:, :3, 0], out[:, :3, 1], out[:, :3, 2] = z, x, y  # approach, closing, normal
    out[:, :3, 3] = grasps[:, :3, 3] + (GRIPPER_DEPTH_M + extra_depth_m) * z
    return out


def nms(grasps: np.ndarray, scores: np.ndarray, dist_m: float = 0.02, angle_deg: float = 30.0) -> np.ndarray:
    """Greedy NMS on AnyGrasp-convention poses; returns kept indices, best score first."""
    order = np.argsort(-scores)
    keep: list[int] = []
    cos_thr = np.cos(np.deg2rad(angle_deg))
    for i in order:
        ok = True
        for j in keep:
            if np.linalg.norm(grasps[i, :3, 3] - grasps[j, :3, 3]) < dist_m:
                same_approach = float(grasps[i, :3, 0] @ grasps[j, :3, 0]) > cos_thr
                # a two-finger grasp is symmetric under a 180 deg turn about its approach
                same_closing = abs(float(grasps[i, :3, 1] @ grasps[j, :3, 1])) > cos_thr
                if same_approach and same_closing:
                    ok = False
                    break
        if ok:
            keep.append(int(i))
    return np.asarray(keep, dtype=int)


def _overlay(rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, grasps: np.ndarray, widths: np.ndarray) -> bytes:
    """The object mask tinted, and each grasp drawn as its finger line and approach (best first)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    img[mask] = (0.6 * img[mask] + 0.4 * np.array([0, 180, 255])).astype(np.uint8)

    def px(p: np.ndarray) -> tuple[int, int] | None:
        if p[2] <= 1e-6:
            return None
        uv = K @ (p / p[2])
        return int(round(uv[0])), int(round(uv[1]))

    for k in range(len(grasps) - 1, -1, -1):
        c, a, b = grasps[k, :3, 3], grasps[k, :3, 0], grasps[k, :3, 1]
        w = float(widths[k]) if k < len(widths) else 0.08
        color = (0, 255, 0) if k == 0 else (255, 200, 0)
        p1, p2, p3 = px(c - w / 2 * b), px(c + w / 2 * b), px(c - 0.06 * a)
        if p1 and p2:
            cv2.line(img, p1, p2, color, 2)
        if p3 and px(c):
            cv2.line(img, p3, px(c), color, 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _plan(req: PlanRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
    if _ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    rgb = _b64_to_np(req.rgb_base64).astype(np.uint8)
    depth = _b64_to_np(req.depth_base64).astype(np.float32)
    K = _b64_to_np(req.cam_K_base64).astype(np.float64)
    segmap = _b64_to_np(req.segmap_base64).astype(np.int32)
    if rgb.shape[:2] != depth.shape[:2] or segmap.shape[:2] != depth.shape[:2]:
        raise HTTPException(status_code=400, detail="RGB, depth, and segmap must have matching height/width")
    seg = np.where(segmap == req.segmap_id, 1, 0).astype(np.int32)
    mask = seg.astype(bool)
    empty = (np.empty((0, 4, 4)), np.empty((0,)), np.empty((0,)))
    if not mask.any():
        logger.warning("empty object mask")
        return (*empty, _overlay(rgb, mask, K, empty[0], empty[2]))
    z_range = req.z_range or [1e-6, 1.5]
    t0 = time.perf_counter()
    with _LOCK:
        pc_full, pc_segments, _ = _ESTIMATOR.extract_point_clouds(
            depth, K, segmap=seg, rgb=rgb, z_range=z_range, skip_border_objects=False
        )
        seg_key = next((k for k in pc_segments if int(k) == 1), None)
        if seg_key is None or len(pc_segments[seg_key]) == 0:
            logger.warning("no object points inside z_range %s", z_range)
            return (*empty, _overlay(rgb, mask, K, empty[0], empty[2]))
        pred, scores, _contacts, openings = _ESTIMATOR.predict_scene_grasps(
            pc_full, pc_segments={seg_key: pc_segments[seg_key]}, local_regions=True,
            filter_grasps=True, forward_passes=_FORWARD_PASSES,
        )  # fmt: skip
    g = np.asarray(pred.get(seg_key, np.empty((0, 4, 4))), dtype=np.float64).reshape(-1, 4, 4)
    s = np.asarray(scores.get(seg_key, np.empty((0,))), dtype=np.float64).reshape(-1)
    w = np.asarray(openings.get(seg_key, np.full(len(s), 0.08)), dtype=np.float64).reshape(-1)
    n_raw = len(s)
    if n_raw == 0:
        logger.warning("Contact-GraspNet returned no grasps")
        return (*empty, _overlay(rgb, mask, K, empty[0], empty[2]))
    c = np.asarray(_contacts.get(seg_key, np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    if len(c) == len(g):  # where the predicted contacts lie along each grasp's approach, from its origin
        along = np.einsum("ij,ij->i", c - g[:, :3, 3], g[:, :3, 2])
        logger.info("contact depth along approach: mean=%.4f min=%.4f max=%.4f (model frame; fingertip baseline %.4f); "
                    "openings mean=%.4f", along.mean(), along.min(), along.max(), GRIPPER_DEPTH_M, w.mean())  # fmt: skip
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(pc_segments[seg_key])[:, :3]).query(c)
        logger.info("contact to nearest object point: median=%.4f max=%.4f m", float(np.median(d)), float(d.max()))
    g = cgn_to_anygrasp(g, _EXTRA_DEPTH_M)
    if _TOP_DOWN:
        top = g[:, 2, 0] >= _TOP_DOWN_COS  # approach . camera viewing direction (+z)
        if top.any():
            g, s, w = g[top], s[top], w[top]
    keep = nms(g, s)[: max(1, int(req.max_grasps))]
    g, s, w = g[keep], s[keep], w[keep]
    logger.info(
        "grasps: raw=%d kept=%d best=%.3f object_points=%d (%.2fs)",
        n_raw, len(s), float(s[0]), len(pc_segments[seg_key]), time.perf_counter() - t0,
    )  # fmt: skip
    return g, s, w, _overlay(rgb, mask, K, g, w)


@app.post("/plan", response_model=PlanResponse)
def plan_endpoint(req: PlanRequest) -> PlanResponse:
    g, s, w, _ = _plan(req)
    return PlanResponse(grasps_base64=_np_to_b64(g), scores_base64=_np_to_b64(s), widths_base64=_np_to_b64(w),
                        n_grasps=int(len(s)), best_score=float(s[0]) if len(s) else None)  # fmt: skip


@app.post("/plan_viz", response_model=PlanVizResponse)
def plan_viz_endpoint(req: PlanRequest) -> PlanVizResponse:
    g, s, w, overlay = _plan(req)
    return PlanVizResponse(
        grasps_base64=_np_to_b64(g), scores_base64=_np_to_b64(s), widths_base64=_np_to_b64(w),
        n_grasps=int(len(s)), best_score=float(s[0]) if len(s) else None,
        overlay_jpeg_base64=base64.b64encode(overlay).decode(), grasp_thumbnail_jpeg_base64=[],
    )  # fmt: skip


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "backend": "contact_graspnet", "model_loaded": _ESTIMATOR is not None,
            "checkpoint_path": _CKPT, "top_down_grasp": _TOP_DOWN, "extra_depth_m": _EXTRA_DEPTH_M,
            "detection_only": True}  # fmt: skip


@app.post("/reset_state")
def reset_state() -> dict[str, Any]:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True, "backend": "contact_graspnet", "model_loaded": _ESTIMATOR is not None}


def load_estimator(cgn_root: Path, ckpt_dir: str, forward_passes: int) -> Any:
    """Contact-GraspNet's GraspEstimator with the checkpoint's weights, on the GPU."""
    import torch

    root = str(cgn_root)
    for p in (root, os.path.join(root, "Pointnet_Pointnet2_pytorch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    ckpt = cgn_root / ckpt_dir
    cfg = config_utils.load_config(str(ckpt), batch_size=forward_passes, arg_configs=[])
    est = GraspEstimator(cfg)
    # torch>=2.6 loads weights_only by default; the checkpoint also pickles numpy scalars.
    safe = [np.core.multiarray.scalar, np.dtype, *{type(np.dtype(t)) for t in ("float64", "float32", "int64")}]
    with torch.serialization.safe_globals(safe):
        state = torch.load(ckpt / "checkpoints" / "model.pt", map_location="cpu", weights_only=True)
    est.model.load_state_dict(state["model"])
    est.model.eval()
    return est


def main() -> None:
    global _ESTIMATOR, _CKPT, _FORWARD_PASSES, _TOP_DOWN, _EXTRA_DEPTH_M

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cgn-root", type=Path, default=Path(os.environ.get("CGN_ROOT", "~/repos/contact_graspnet_pytorch")).expanduser())
    ap.add_argument("--ckpt-dir", default="checkpoints/contact_graspnet")
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--forward-passes", type=int, default=2, help="more passes sample more contacts")
    ap.add_argument("--extra-depth", type=float, default=_EXTRA_DEPTH_M,
                    help="metres past the predicted contacts along the approach (AnyGrasp-like depth)")
    ap.add_argument("--top-down", dest="top_down", action="store_true", default=True)
    ap.add_argument("--no-top-down", dest="top_down", action="store_false")
    a = ap.parse_args()
    _FORWARD_PASSES, _TOP_DOWN, _EXTRA_DEPTH_M = a.forward_passes, a.top_down, a.extra_depth
    logger.info("loading Contact-GraspNet from %s", a.cgn_root / a.ckpt_dir)
    _ESTIMATOR = load_estimator(a.cgn_root, a.ckpt_dir, a.forward_passes)
    _CKPT = str(a.cgn_root / a.ckpt_dir)
    logger.info("Contact-GraspNet ready on %s:%d (AnyGrasp API)", a.host, a.port)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
