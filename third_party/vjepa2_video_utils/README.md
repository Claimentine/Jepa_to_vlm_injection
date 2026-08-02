# vjepa2 video utils (vendored)

`transforms.py`, `volume_transforms.py`, `functional.py`, `randaugment.py` are
copied unmodified from Meta's public V-JEPA2 release
(`vjepa2/src/datasets/utils/video/`). The public ThinkJEPA release trimmed
this subdirectory, which breaks `load_dense_jepa_encoder` in
`cache_train/thinker_train.py` (it imports from `datasets.utils.video.*`
transitively). Vendoring these four files here is the smallest fix; no
ThinkJEPA-internal code is duplicated.

If a future ThinkJEPA release restores these files, point `THINKJEPA_ROOT`
at that checkout and this vendored copy can be dropped in favor of the
upstream one.
