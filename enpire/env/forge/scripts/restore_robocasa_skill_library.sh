#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Restore ENPIRE's RoboCasa seed skill library into this checkout, LOCAL ONLY.
#
# The library was removed upstream in d81ae78 ("Legal remediation"), so it is not
# redistributed here: this script copies it out of this repository's own git
# history into a git-ignored directory. Do not commit or publish the result.
# groot_only.py (a GR00T rollout wrapper) is skipped: GR00T is not part of this build.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REV="d81ae78^"
SUB="enpire/env/forge/cap/saved_scripts/robocasa_skill_library"
if ! git -C "$ROOT" cat-file -e "${REV}:${SUB}" 2>/dev/null; then
    echo "error: ${REV} is not in this clone's history (shallow clone?). Run: git -C $ROOT fetch --unshallow" >&2
    exit 1
fi
rm -rf "${ROOT:?}/${SUB}"
git -C "$ROOT" archive "$REV" "$SUB" | tar -x -C "$ROOT"
rm -f "$ROOT/$SUB/groot_only.py"
echo "restored $(find "$ROOT/$SUB" -name '*.py' | wc -l | tr -d ' ') files into $SUB (git-ignored, local only)"
