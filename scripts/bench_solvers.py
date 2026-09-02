"""Micro-benchmark of the disambiguation solvers on real candidate pools.

Timings on synthetic pools mislead: pools drawn from short random strings over a
small alphabet are far denser in prefix relations than a language model's top-k,
which changes which solver wins.  So this records the pools an actual generation
run encounters, then times each solver on them repeatedly.

  python scripts/bench_solvers.py --lang en --topks 8 16 32 64 128 --n 6
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mwis_stego.antichain import exact, exact_trie, greedy
from mwis_stego.coder import CoderConfig, StepStats, encode
from mwis_stego.data import SOURCES
from mwis_stego.model import StegoLM

SOLVERS = {"exact": exact, "exact_trie": exact_trie, "greedy": greedy}


def collect(lm, prompts, topk, args):
    pools = []
    rng = random.Random(args.seed)
    for p in prompts:
        message = [rng.randint(0, 1) for _ in range(args.message_bits)]
        cfg = CoderConfig(precision=26, topk=topk, method="exact",
                          max_tokens=args.max_tokens, device=lm.device)
        stats = StepStats()
        stats.record_pools = True
        encode(lm.model, lm.vocab, message, lm.chat_context(p.text), cfg,
               lm.banned_ids, stats=stats)
        pools.extend(stats.pools)
    return pools


def time_solver(pools, solver, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for toks, w in pools:
            solver(toks, w)
        best = min(best, time.perf_counter() - t0)   # min: least contaminated by noise
    return 1e6 * best / max(len(pools), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--topks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--source", default="flores", choices=list(SOURCES))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--message-bits", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    lm = StegoLM(args.model, device=args.device)
    prompts = SOURCES[args.source](args.lang, n=args.n)

    print(f"{'topk':>5} {'pools':>7} {'conflict%':>10} "
          + " ".join(f"{n:>12}" for n in SOLVERS) + f" {'speedup':>9}")
    print("-" * 78)
    for topk in args.topks:
        pools = collect(lm, prompts, topk, args)
        # A pool is conflict-free when nothing is removed from it.
        conflicted = sum(1 for toks, w in pools if len(exact(toks, w)) < len(toks))
        us = {n: time_solver(pools, s, args.repeats) for n, s in SOLVERS.items()}
        print(f"{topk:5d} {len(pools):7d} {100*conflicted/max(len(pools),1):9.1f}% "
              + " ".join(f"{us[n]:12.1f}" for n in SOLVERS)
              + f" {us['exact_trie']/us['exact']:8.2f}x")
    print("\nmicroseconds per pool; `speedup` is exact_trie / exact")


if __name__ == "__main__":
    main()
