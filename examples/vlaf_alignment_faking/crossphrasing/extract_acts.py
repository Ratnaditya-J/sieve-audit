#!/usr/bin/env python3
"""
Extract last-token residual-stream activations for the cross-phrasing variants.
Runs ON a GPU pod. No generation, no judges — just forward passes capturing the
last-token hidden state at every layer, so the local analysis can pick the
paper's layer (15) and also sweep neighbors.

Output: one compressed .npz per variant with
  acts   : float16 [n, n_layers+1, d]   (hidden_states[layer][:, -1, :])
  labels : int8    [n]                  (0=monitored, 1=unmonitored)
  family : list[str]
  pid    : list[str]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-glob", default="prompts/*.jsonl")
    ap.add_argument("--out-dir", default="acts")
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={dev} model={args.model}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, output_hidden_states=True
    ).to(dev)
    model.eval()

    files = sorted(glob.glob(args.prompts_glob))
    print(f"[extract] {len(files)} variant files: {[os.path.basename(f) for f in files]}", file=sys.stderr)

    for fpath in files:
        variant = os.path.splitext(os.path.basename(fpath))[0]
        rows = [json.loads(l) for l in open(fpath) if l.strip()]
        texts, labels, fams, pids = [], [], [], []
        for r in rows:
            msgs = json.loads(r["text"])
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            texts.append(prompt)
            labels.append(int(r["label"]))
            fams.append(r["family"])
            pids.append(r["prompt_id"])

        all_acts = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len, add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = model(**enc)
            # hidden_states: tuple (n_layers+1) of [B, T, d]; left-padded => last token at -1
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # [B, L+1, d]
            all_acts.append(hs.to(torch.float16).cpu().numpy())
            print(f"[extract] {variant}: {min(i+args.batch_size, len(texts))}/{len(texts)}", file=sys.stderr)
        acts = np.concatenate(all_acts, axis=0)  # [n, L+1, d]
        outp = os.path.join(args.out_dir, f"{variant}_acts.npz")
        np.savez_compressed(outp, acts=acts, labels=np.array(labels, dtype=np.int8),
                            family=np.array(fams), pid=np.array(pids))
        print(f"[extract] saved {outp}  acts.shape={acts.shape}", file=sys.stderr)

    print("[extract] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
