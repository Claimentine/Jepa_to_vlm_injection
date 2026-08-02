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
EXTRACTOR="$THINKJEPA_ROOT/cache_train/qwen3_cache_extractor.py"
if [ -f "$EXTRACTOR" ] && grep -q "return p.parse_thinker_cache_extraction_args()" "$EXTRACTOR"; then
  sed -i 's/return p\.parse_thinker_cache_extraction_args()/return p.parse_args()/' "$EXTRACTOR"
  echo "[OK] patched $EXTRACTOR (parse_args recursion bug)"
fi
