"""A1 baseline (per README's "Planned additional experiments" A1 row):
"VLM-only, trained" -- a trained adapter/head on the frozen VLM's OWN hidden
states, with ZERO external (JEPA) conditioning and NO perturbation of the
VLM's forward pass at all (no decoder hook, no injected tensor of any kind).
This is what ThinkJEPA's own "Qwen3-VL Thinking" baseline row in their Table 1
actually is (a *trained* downstream head on VLM features, not zero-shot) --
stronger and fairer than this project's `zeroshot_none` row (--injection none
in train_qa_full.py / eval_forward_checkpoint.py's NONE sentinel), which has
no training at all.

Distinct from train_qa_full.py's existing --injection {random,constant}
content-free controls: those KEEP the real injection mechanism (the same
cross_attn/film hook perturbing the frozen decoder's hidden_states at 4
middle layers), just feed it content-free input, so they isolate "does the
injection MECHANISM's extra capacity help, even without real content" from
"does real JEPA content help". This script isolates a different question:
"does ANY supervised adaptation on this task help at all, with the VLM's own
forward computation completely untouched" -- i.e. whether merely reading
(not rewriting) the VLM's existing representations and learning a small
scoring head on top already buys some accuracy over true zero-shot, for
reasons having nothing to do with cross-modal injection.

Reuses train_qa_full.py's helpers (build_option_block, build_prompt_text,
build_inputs, append_answer_tokens, AccTally) unmodified via import, same
qa_split schema, same val/save cadence conventions, so results are directly
comparable to job-13 (true independent jepa x cross_attn) and job-145 (the
bridge-depth-vs-joint ablation) at the same --max_steps budget.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import train_qa_full as fwd  # noqa: E402


class VLMOnlyHead(nn.Module):
    """Reads a mean-pooled hidden state from the frozen VLM's own last
    decoder layer and outputs a scalar score adjustment. Zero-initialized
    final layer (same convention as LayerWiseJEPAInjector's gates and
    TokenTrajectoryProbe's output layer elsewhere in this project): at
    step 0 this is an exact no-op, so the run starts IDENTICAL to the pure
    zero-shot baseline and can only improve on it by actually learning
    something from supervised training on this task -- not from some
    lucky random initialization bias."""

    def __init__(self, hidden_size, mlp_hidden=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled):
        return self.mlp(pooled).squeeze(-1)


def candidate_logprob_with_head(model, model_inputs, tokenizer, letter, head):
    """Same scoring as train_qa_full.py's candidate_logprob (log P(letter |
    video, question, options)), PLUS head's own scalar output added on top --
    computed from the SAME forward call (output_hidden_states=True), not a
    second pass, so the VLM's forward computation is identical to the plain
    zero-shot path; only this function's own bookkeeping differs."""
    prefix_len = model_inputs["input_ids"].shape[1]
    input_ids, attn_mask, suffix_ids = fwd.append_answer_tokens(
        tokenizer, model_inputs["input_ids"], model_inputs["attention_mask"], letter
    )
    call_inputs = dict(model_inputs)
    call_inputs["input_ids"] = input_ids
    call_inputs["attention_mask"] = attn_mask
    if "mm_token_type_ids" in call_inputs and torch.is_tensor(call_inputs["mm_token_type_ids"]):
        tt = call_inputs["mm_token_type_ids"]
        pad = torch.zeros((tt.shape[0], suffix_ids.shape[1]), dtype=tt.dtype, device=tt.device)
        call_inputs["mm_token_type_ids"] = torch.cat([tt, pad], dim=1)
    out = model(**call_inputs, output_hidden_states=True)
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + suffix_ids.shape[1], device=out.logits.device)
    token_logprobs = out.logits[0, positions].float().log_softmax(dim=-1)
    targets = suffix_ids[0].to(out.logits.device)
    base_logprob = token_logprobs.gather(1, targets.unsqueeze(1)).sum()
    # out.hidden_states[-1]: (1, seq_len, hidden_size) -- the frozen model's
    # OWN final-layer representation, mean-pooled over the full sequence
    # (prompt + video tokens + appended answer letter). No pooling-over-time
    # structure needed here (unlike JEPA's per-frame tokens): this is a
    # single scalar summary of "what the VLM itself already represents".
    pooled = out.hidden_states[-1][0].float().mean(dim=0)
    head_score = head(pooled.unsqueeze(0))[0]
    return base_logprob + head_score


def evaluate_with_head(model, processor, head, items, device, rng):
    model.eval()
    tally = fwd.AccTally()
    seen_exc_types = set()
    with torch.no_grad():
        for item in items:
            try:
                option_lines, correct_letter = fwd.build_option_block(item, rng)
                prompt_text = fwd.build_prompt_text(item["question"], option_lines)
                inputs = fwd.build_inputs(processor, item["in_clip"], prompt_text)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                scores = torch.stack([
                    candidate_logprob_with_head(model, model_inputs, processor.tokenizer, "A", head),
                    candidate_logprob_with_head(model, model_inputs, processor.tokenizer, "B", head),
                ])
                tally.add(item, ("A" if scores[0] > scores[1] else "B") == correct_letter)
            except Exception as e:
                exc_name = type(e).__name__
                print(f"[WARN] eval: skipping item {item.get('pid')}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
    model.train()
    return tally


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=os.path.join(fwd.BASE, "raw_data/qa_split_temporal.json"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000)
    ap.add_argument("--max_val_items", type=int, default=200)
    ap.add_argument("--max_train_items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.split) as f:
        split = json.load(f)
    train_items = [it for it in split["train"] if fwd.has_features(it)]
    val_items = [it for it in split["val"] if fwd.has_features(it)][: args.max_val_items]
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    print(f"[INFO] usable (features present): train={len(train_items)} val={len(val_items)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(fwd.MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(fwd.MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size

    head = VLMOnlyHead(hidden_size).to(device).float()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    print(f"[INFO] A1 (VLM-only, trained): trainable_params={sum(p.numel() for p in head.parameters()):,} "
          f"(head only -- VLM forward pass is completely unperturbed, no hook, no injection)", flush=True)

    step = 0
    n_skipped = 0
    seen_exc_types = set()
    best_val_acc = -1.0
    log_f = open(os.path.join(args.out_dir, "train_log.jsonl"), "w")

    for epoch in range(args.epochs):
        rng.shuffle(train_items)
        for item in train_items:
            try:
                option_lines, correct_letter = fwd.build_option_block(item, rng)
                prompt_text = fwd.build_prompt_text(item["question"], option_lines)
                inputs = fwd.build_inputs(processor, item["in_clip"], prompt_text)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                scores = torch.stack([
                    candidate_logprob_with_head(model, model_inputs, processor.tokenizer, "A", head),
                    candidate_logprob_with_head(model, model_inputs, processor.tokenizer, "B", head),
                ]).unsqueeze(0)
                target = torch.tensor([0 if correct_letter == "A" else 1], device=device)
                loss = nn.functional.cross_entropy(scores, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            except Exception as e:
                n_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] skipping item {item.get('pid')}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
                continue

            step += 1
            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}", flush=True)
            log_f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item()}) + "\n")
            log_f.flush()

            if step % args.val_every_steps == 0:
                val_tally = evaluate_with_head(model, processor, head, val_items, device, rng)
                print(f"[epoch {epoch} step {step}] val_acc(logprob)={val_tally.summary_str()}", flush=True)
                log_f.write(json.dumps({
                    "step": step, "val_acc": val_tally.acc, "val_acc_by_difficulty": val_tally.acc_by_difficulty,
                }) + "\n")
                log_f.flush()
                if val_tally.acc > best_val_acc:
                    best_val_acc = val_tally.acc
                    torch.save({"args": vars(args), "head_state": head.state_dict(),
                                "step": step, "val_acc": val_tally.acc,
                                "val_acc_by_difficulty": val_tally.acc_by_difficulty},
                               os.path.join(args.out_dir, "best.pt"))

            if args.save_every_steps and step % args.save_every_steps == 0:
                torch.save({"args": vars(args), "head_state": head.state_dict(), "step": step},
                           os.path.join(args.out_dir, f"step_{step}.pt"))
                print(f"[epoch {epoch} step {step}] saved checkpoint -> step_{step}.pt", flush=True)

            if args.max_steps and step >= args.max_steps:
                print(f"[INFO] --max_steps {args.max_steps} reached, stopping", flush=True)
                break
        if args.max_steps and step >= args.max_steps:
            break

    summary = {"done": True, "best_val_acc": best_val_acc, "steps_completed": step, "items_skipped": n_skipped}
    print(f"[DONE] A1 (VLM-only, trained)  best_val_acc(logprob)={best_val_acc:.4f}  "
          f"steps_completed={step}  items_skipped={n_skipped}", flush=True)
    log_f.write(json.dumps(summary) + "\n")
    log_f.flush()
    log_f.close()


if __name__ == "__main__":
    main()
