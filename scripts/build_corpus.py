"""Build paired stego / cover corpora for steganalysis.

The negative class is *not* human text. Arithmetic coding driven by a uniformly
random message samples from the renormalised retained pool, so generating with
`--methods none` samples from the model's own top-k pool with no disambiguation
applied. Cover and stego therefore share the model, the prompt, the top-k cutoff,
the temperature, the probability quantisation and the token count, and differ in
exactly one thing: whether the pool was disambiguated.

That makes detection accuracy a direct test of the paper's thesis. A solver that
retains more probability mass leaves the two distributions closer together, so a
higher eta should show up as a detector that separates them less well. Comparing
against human sentences instead would fold in every difference between the model
and real text, which swamps the effect being measured.

Every text is generated to exactly `--tokens` tokens by supplying a message far
longer than the budget, so the two classes cannot be told apart by length.

  python scripts/build_corpus.py --langs en zh ja --methods exact greedy \
      --topks 32 --n 400 --tokens 96 --out runs/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mwis_stego.antichain import EnumerationInfeasible
from mwis_stego.coder import CoderConfig, encode
from mwis_stego.data import SOURCES
from mwis_stego.model import StegoLM


def generate(lm, prompt, method, topk, args, rng):
    """One text of exactly args.tokens tokens, or None if the solver gives up."""
    # Far more bits than the token budget can absorb, so the budget always binds
    # and every sample has the same length.
    message = [rng.randint(0, 1) for _ in range(args.tokens * 16)]
    cfg = CoderConfig(precision=args.precision, topk=topk, temp=args.temp,
                      method=method, max_tokens=args.tokens, device=lm.device)
    ctx = lm.chat_context(prompt.text)
    try:
        ids, stego, stats = encode(lm.model, lm.vocab, message, ctx, cfg, lm.banned_ids)
    except EnumerationInfeasible:
        return None
    return {
        "text": stego.decode("utf-8", errors="replace"),
        "tokens": len(ids),
        "mean_eta": stats.summary()["mean_eta"],
        "kld_c_bits": stats.summary()["kld_c_bits"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "ja"])
    ap.add_argument("--methods", nargs="+", default=["exact", "greedy"],
                    help="stego classes; the cover class is always `none`")
    ap.add_argument("--topks", nargs="+", type=int, default=[32])
    ap.add_argument("--source", default="flores", choices=list(SOURCES))
    ap.add_argument("--n", type=int, default=400, help="prompts per (lang, topk)")
    ap.add_argument("--tokens", type=int, default=96, help="exact length of every text")
    ap.add_argument("--precision", type=int, default=26)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/corpus.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lm = StegoLM(args.model, device=args.device)
    prompts = {l: SOURCES[args.source](l, n=args.n) for l in args.langs}
    classes = list(args.methods) + ["none"]

    n_written = 0
    with out.open("w", encoding="utf-8") as fh:
        for lang in args.langs:
            for topk in args.topks:
                for method in classes:
                    # Same seed per class: each class sees the same message bits
                    # against the same prompts, so the classes differ only in the
                    # disambiguation applied.
                    rng = random.Random(args.seed)
                    kept = 0
                    for p in prompts[lang]:
                        rec = generate(lm, p, method, topk, args, rng)
                        if rec is None:
                            continue
                        rec.update({
                            "lang": lang, "topk": topk, "method": method,
                            "key": p.key,
                            "label": 0 if method == "none" else 1,
                        })
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept += 1
                    fh.flush()
                    n_written += kept
                    print(f"{lang} topk={topk} {method:9} {kept:5d} texts", flush=True)
    print(f"\nwrote {n_written} texts to {out}")


if __name__ == "__main__":
    main()
