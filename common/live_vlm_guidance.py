"""Live (gradient-carrying) VLM guidance extraction, for training scripts
that want to fine-tune the VLM itself (via LoRA) instead of reading
pre-cached, frozen vlm_old/vlm_new features from vlm_guidance_cache_paired/
{pid}.npz (as extract_vlm_guidance_paired.py produces and
train_latent_world_model_full.py/train_joint_coupled_normalized.py/
finetune_egodex_predictor.py currently consume).

This is the SAME decode + teacher-forced-inference logic as
extract_vlm_guidance_paired.py's decode_clip_with_timeout/
infer_qwen_teacher_forced, duplicated here (not imported) because that
script hardcodes torch.inference_mode() around its one forward call --
correct for a one-shot, frozen-VLM extraction pass, but inference_mode
tensors can NEVER be used in autograd afterwards (not even by a later,
separate backward() call outside that context), so a live/trainable VLM
needs the forward call made WITHOUT that wrapper. Keeping
extract_vlm_guidance_paired.py itself untouched avoids any risk to the
already-validated cache-generation pipeline.

Per explicit user request (2026-11-07/08): "fine-tune all modules, only
JEPA's own encoder stays frozen" -- V-JEPA2's encoder output
(vjepa_input_feats/vjepa_target_feats) stays read from the existing cache
unchanged (the encoder is frozen and deterministic given the input frames,
so caching it is still exactly correct, not a shortcut); only the VLM side
of the cache (vlm_old/vlm_new) is replaced by a live, gradient-carrying
forward pass here.
"""
import os
import sys

import numpy as np
import torch

THINKJEPA_ROOT = os.environ.get("THINKJEPA_ROOT", "/home/jovyan/ThinkJEPA")
if THINKJEPA_ROOT not in sys.path:
    sys.path.insert(0, THINKJEPA_ROOT)

import multiprocessing as mp  # noqa: E402
_MP_CTX = mp.get_context("spawn")

DECODE_TIMEOUT_S = 60  # matches extract_vlm_guidance_paired.py's own constant


def _decode_clip_worker(conn, video_path_str, num_frames):
    try:
        from decord import VideoReader, cpu
        from cache_train.rebuild_causal_cache import uniform_indices

        reader = VideoReader(video_path_str, ctx=cpu(0), num_threads=1)
        total = int(len(reader))
        if total < 1:
            raise ValueError(f"no frames in {video_path_str}")
        fps = float(reader.get_avg_fps())
        indices = uniform_indices(0, total, num_frames)
        frames = np.asarray(reader.get_batch(indices).asnumpy())
        frames = np.ascontiguousarray(frames[..., :3].astype(np.uint8, copy=False))
        try:
            timestamps = np.asarray(reader.get_frame_timestamp(indices), dtype=np.float64)
            centers = (timestamps[:, 0] + timestamps[:, 1]) * 0.5
        except Exception:
            fps_safe = fps if fps > 0 else 30.0
            centers = indices.astype(np.float64) / fps_safe
        conn.send(("ok", (frames, centers, fps)))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def decode_clip_with_timeout(video_path, num_frames):
    """Returns (frames uint8 [num_frames,H,W,3], frame_center_times, source_fps).
    Identical to extract_vlm_guidance_paired.py's own function -- see that
    module's docstring for why this runs in a spawned subprocess (CUDA
    context already initialized in the parent by the time any clip is
    decoded; forking after CUDA init hangs at child startup)."""
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_decode_clip_worker, args=(child_conn, str(video_path), num_frames))
    proc.start()
    child_conn.close()
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
        raise TimeoutError(f"video decode exceeded {DECODE_TIMEOUT_S}s on {video_path}")
    if status == "err":
        raise payload
    return payload


def infer_qwen_teacher_forced_live(args, model, processor, saved, device, past_frames, past_times, in_fps,
                                    question, correct_caption):
    """Same as extract_vlm_guidance_paired.py's infer_qwen_teacher_forced,
    EXCEPT the forward call is NOT wrapped in torch.inference_mode() -- the
    whole point here is a gradient-carrying pass through model's own (LoRA-
    wrapped) trainable parameters. Caller must NOT wrap this in torch.no_grad()
    either when training (do that only for a true eval pass)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from cache_train.rebuild_causal_cache import _effective_sample_fps

    for values in saved.values():
        values.clear()

    effective_fps = _effective_sample_fps(past_times, in_fps)
    exact_past_pil = [Image.fromarray(frame) for frame in past_frames]
    video_content = {
        "type": "video",
        "video": exact_past_pil,
        "resized_height": int(args.qwen_res),
        "resized_width": int(args.qwen_res),
        "sample_fps": effective_fps,
        "raw_fps": effective_fps,
    }
    messages = [{"role": "user", "content": [video_content, {"type": "text", "text": str(question)}]}]
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )
    if not videos or len(videos) != 1:
        raise RuntimeError("Qwen vision processing did not return exactly one video")
    video_tensors, video_metadata = [], []
    for value in videos:
        if isinstance(value, tuple):
            video_tensors.append(value[0])
            video_metadata.append(value[1])
        else:
            video_tensors.append(value)
    if int(video_tensors[0].shape[0]) != past_frames.shape[0]:
        raise AssertionError(
            f"Qwen vision processing changed the exact past-frame count: {tuple(video_tensors[0].shape)}"
        )

    rendered_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {
        "text": [rendered_text], "videos": video_tensors, "return_tensors": "pt",
        "padding": True, "do_resize": False,
    }
    if images:
        processor_kwargs["images"] = images
    if video_metadata:
        processor_kwargs["video_metadata"] = video_metadata
    processor_kwargs.update(video_kwargs or {})
    prompt_inputs = processor(**processor_kwargs)
    prefix_len = prompt_inputs["input_ids"].shape[1]

    answer_ids = processor.tokenizer(
        f" {correct_caption}", add_special_tokens=False, return_tensors="pt",
    )["input_ids"]
    full_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(answer_ids)], dim=1)

    call_inputs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
    call_inputs["input_ids"] = full_ids
    call_inputs["attention_mask"] = full_mask
    call_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in call_inputs.items()}

    model(**call_inputs)  # NOT wrapped in no_grad/inference_mode -- see docstring

    layer_states = []
    for layer_idx in args.layers:
        calls = saved[f"dec_{layer_idx}"]
        if len(calls) != 1:
            raise RuntimeError(f"expected exactly one forward call per item, got {len(calls)} for layer {layer_idx}")
        layer_states.append(calls[0][0])  # [1, full_len, D]
    all_states = torch.stack(layer_states, dim=0)  # [L, full_len, D]
    vlm_old = all_states[:, :prefix_len, :]
    vlm_new = all_states[:, prefix_len:, :]
    return vlm_old, vlm_new


def load_qwen_with_lora(args, device, lora_r=8, lora_alpha=16, lora_dropout=0.0):
    """Loads Qwen3-VL + registers the same decoder-hidden-state hooks
    extract_vlm_guidance_paired.py uses (via cache_train.rebuild_causal_cache.
    load_qwen), then LoRA-wraps every nn.Linear in it (301 on Qwen3-VL-2B-
    Thinking, confirmed via job-150's module dump -- no nn.MultiheadAttention
    anywhere) and freezes everything else. Returns (model, processor, saved,
    qwen_sha, wrapped_names)."""
    from cache_train.rebuild_causal_cache import load_qwen
    from common.lora import apply_lora_to_all_layers, freeze_all_except_lora, lora_trainable_param_count

    model, processor, saved, qwen_sha = load_qwen(args, device)
    wrapped = apply_lora_to_all_layers(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    freeze_all_except_lora(model)
    n_trainable = lora_trainable_param_count(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] load_qwen_with_lora: wrapped {len(wrapped)} Linear layers "
          f"(r={lora_r} alpha={lora_alpha}): trainable={n_trainable:,}/{n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)", flush=True)
    return model, processor, saved, qwen_sha, wrapped
