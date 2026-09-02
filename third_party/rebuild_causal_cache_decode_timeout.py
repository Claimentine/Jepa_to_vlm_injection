"""Subprocess-isolated, hard-timeout wrapper around
cache_train.rebuild_causal_cache.inspect_and_decode_video(), so a hung
per-clip decode/frame-selection call can be killed and skipped without
losing the parent process's already-loaded Qwen3-VL/V-JEPA2 models -- a
whole-process restart (the previous mitigation) forces reloading both,
at ~30-60min cost, just to skip one clip.

Confirmed 2026-09-02: clip -9oqHVK5-5c/598.mp4 hangs deterministically at
inspect_and_decode_video()'s call site inside rebuild_causal_cache.py's
main() loop (0% GPU utilization, no forward progress) on every attempt,
despite decoding and frame-selecting fine when the exact same call is
reproduced in an isolated fresh process outside that loop. The cause is
unknown, but a hard subprocess timeout bounds the damage regardless of
cause, the same way vans_qa/full_scale/train_qa_full.py's DECODE_TIMEOUT_S
already does for the sibling forward-direction pipeline.

spawn, not fork: by the time this is ever called, the parent process has
already initialized a CUDA context for two loaded models. Forking after
CUDA init inherits driver state the child can't safely reuse and hangs at
child startup rather than doing real decode work -- see
train_qa_full.py's identical _MP_CTX pattern, which hit exactly this
failure mode first and is why this file follows the same shape.
"""
import multiprocessing as mp
import os
import sys
from pathlib import Path

_THINKJEPA_ROOT = os.environ.get(
    "THINKJEPA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, _THINKJEPA_ROOT)

# Generous: this only bounds the decode/frame-selection step, not the full
# per-clip pipeline (V-JEPA2 encode + Qwen3-VL generate stay in the parent
# process, unaffected). Observed steady-state cost for the WHOLE per-clip
# pipeline is ~11-30s, so 300s leaves a wide margin before treating
# something as hung rather than just a slow decode.
DECODE_TIMEOUT_S = 300

_MP_CTX = mp.get_context("spawn")


def _decode_worker(conn, video_path_str):
    try:
        from cache_train.rebuild_causal_cache import inspect_and_decode_video
        result = inspect_and_decode_video(Path(video_path_str))
        conn.send(("ok", result))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def inspect_and_decode_video_with_timeout(video_path):
    """Drop-in replacement for inspect_and_decode_video() with a hard
    subprocess timeout. Raises TimeoutError on timeout -- the caller's
    existing per-video try/except in rebuild_causal_cache.py's main() loop
    already logs and skips on any exception, so this integrates at the call
    site alone, no other change needed there.
    """
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_decode_worker, args=(child_conn, str(video_path)))
    proc.start()
    child_conn.close()  # parent's copy of the write end; child still holds its own
    if parent_conn.poll(DECODE_TIMEOUT_S):
        status, payload = parent_conn.recv()
    else:
        status, payload = "timeout", None
    parent_conn.close()
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if status == "timeout":
        raise TimeoutError(f"video decode/frame-selection exceeded {DECODE_TIMEOUT_S}s on {video_path}")
    if status == "err":
        raise payload
    return payload
