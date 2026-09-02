#!/usr/bin/env bash
# The public ThinkJEPA release is missing vjepa2/src/datasets/utils/video/*,
# which load_dense_jepa_encoder() needs transitively. Run this once after
# cloning ThinkJEPA (and before running anything under vans_qa/demo/ or
# vans_qa/full_scale/ that touches V-JEPA2 feature extraction).
#
# Usage: THINKJEPA_ROOT=/path/to/ThinkJEPA ./scripts/patch_thinkjepa.sh
set -euo pipefail

THINKJEPA_ROOT="${THINKJEPA_ROOT:-/projects/bhay/william/ruixin/ThinkJEPA}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$THINKJEPA_ROOT/vjepa2/src/datasets/utils/video"

if [ ! -d "$THINKJEPA_ROOT" ]; then
  echo "error: THINKJEPA_ROOT=$THINKJEPA_ROOT does not exist -- clone ThinkJEPA first" >&2
  exit 1
fi

mkdir -p "$DEST"
cp -v "$REPO_ROOT/third_party/vjepa2_video_utils/"{transforms,volume_transforms,functional,randaugment}.py "$DEST/"
echo "[OK] patched $DEST"

# second, unrelated upstream bug: parse_thinker_cache_extraction_args() calls
# itself instead of calling argparse's p.parse_args() -- breaks
# qwen3_cache_extractor.py (used by extract_vlm_guidance*.py). Never
# upstreamed as a PR to Hai-chao-Zhang/ThinkJEPA -- worth doing so; patched
# locally here in the meantime.
#
# 2026-09-01: this file never actually existed in the public release --
# "qwen3_cache_extractor.py" was a wrong filename baked into our own
# wrapper scripts, not a real ThinkJEPA file. This patch step is a no-op in
# practice (the `[ -f ... ]` guard just skips it) but left in place in case
# it's ever restored upstream under this name.
EXTRACTOR="$THINKJEPA_ROOT/cache_train/qwen3_cache_extractor.py"
if [ -f "$EXTRACTOR" ] && grep -q "return p.parse_thinker_cache_extraction_args()" "$EXTRACTOR"; then
  sed -i 's/return p\.parse_thinker_cache_extraction_args()/return p.parse_args()/' "$EXTRACTOR"
  echo "[OK] patched $EXTRACTOR (parse_args recursion bug)"
fi

# third: give rebuild_causal_cache.py's per-clip decode/frame-selection step
# (inspect_and_decode_video()) a hard subprocess timeout, so one hung clip
# (confirmed 2026-09-02: -9oqHVK5-5c/598.mp4 hangs deterministically, 0% GPU
# util, no forward progress, every attempt) can be skipped without
# restarting the whole process -- which would force reloading both
# Qwen3-VL-2B-Thinking and V-JEPA2 ViT-L, at ~30-60min cost, just to skip
# one clip. See rebuild_causal_cache_decode_timeout.py's docstring for why
# this needs to be a subprocess (spawn, not fork) rather than an in-process
# signal/thread timeout.
REBUILD_SCRIPT="$THINKJEPA_ROOT/cache_train/rebuild_causal_cache.py"
if [ -f "$REBUILD_SCRIPT" ]; then
  cp -v "$REPO_ROOT/third_party/rebuild_causal_cache_decode_timeout.py" \
        "$THINKJEPA_ROOT/cache_train/rebuild_causal_cache_decode_timeout.py"
  if grep -q "selection, past_raw, target_raw = inspect_and_decode_video(video_path)" "$REBUILD_SCRIPT" \
     && ! grep -q "inspect_and_decode_video_with_timeout" "$REBUILD_SCRIPT"; then
    python3 - "$REBUILD_SCRIPT" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
old = "            selection, past_raw, target_raw = inspect_and_decode_video(video_path)\n"
new = (
    "            from cache_train.rebuild_causal_cache_decode_timeout import "
    "inspect_and_decode_video_with_timeout as _iadv_timeout\n"
    "            selection, past_raw, target_raw = _iadv_timeout(video_path)\n"
)
assert text.count(old) == 1, f"expected exactly one match of the call site, found {text.count(old)}"
text = text.replace(old, new)
with open(path, "w") as f:
    f.write(text)
print("[OK] patched", path, "(per-clip decode timeout)")
PYEOF
  else
    echo "[skip] $REBUILD_SCRIPT decode-timeout patch already applied or call site not found"
  fi
fi
