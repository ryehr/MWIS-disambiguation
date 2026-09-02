"""Solver comparison on identical, real candidate pools.

Running each method end to end lets the generated texts diverge, so any
difference in eta mixes the solver's quality with the luck of the sampled text.
This script instead records the actual pools a generation run encounters and
replays all three solvers on the *same* pools, which isolates the solver.

Reported per pool:
  eta      retained probability mass (the quantity -log(eta) = KLD-c penalises)
  time     wall time inside the solver
  status   whether the paper's 2^|C| baseline can run at all

  python scripts/compare_solvers.py --langs en zh ja --topks 8 32 128 --n 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mwis_stego.antichain import EnumerationInfeasible, exact, greedy, enumerate_cc, is_antichain
from mwis_stego.coder import CoderConfig, StepStats, encode
from mwis_stego.data import SOURCES
from mwis_stego.model import StegoLM

SOLVERS = {"exact": exact, "greedy": greedy, "enumerate": enumerate_cc}


def collect_pools(lm, prompts, topk, args):
    pools = []
    rng = random.Random(args.seed)
    for p in prompts:
        message = [rng.randint(0, 1) for _ in range(args.message_bits)]
        cfg = CoderConfig(precision=args.precision, topk=topk, method="exact",
                          max_tokens=args.max_tokens, device=lm.device)
        stats = StepStats()
        stats.record_pools = True
        ctx = lm.chat_context(p.text)
        encode(lm.model, lm.vocab, message, ctx, cfg, lm.banned_ids, stats=stats)
        pools.extend(stats.pools)
    return pools


def evaluate(pools, name, solver):
    total_eta, secs, solved, infeasible, conflicts = 0.0, 0.0, 0, 0, 0
    per_pool = []
    for toks, w in pools:
        total = sum(w) or 1
        t0 = time.perf_counter()
        try:
            kept = solver(toks, w)
        except EnumerationInfeasible:
            infeasible += 1
            per_pool.append(None)
            continue
        secs += time.perf_counter() - t0
        if not is_antichain(toks, kept):
            conflicts += 1
        eta = sum(w[i] for i in kept) / total
        total_eta += eta
        solved += 1
        per_pool.append(eta)
    return {
        "method": name, "pools": len(pools), "solved": solved,
        "infeasible": infeasible, "conflicts": conflicts,
        "mean_eta": total_eta / max(solved, 1),
        "solver_seconds": secs,
        "us_per_pool": 1e6 * secs / max(solved, 1),
        "_per_pool": per_pool,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "ja"])
    ap.add_argument("--topks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--source", default="flores", choices=list(SOURCES))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--message-bits", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--precision", type=int, default=26)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/solvers.jsonl")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lm = StegoLM(args.model, device=args.device)
    prompts = {l: SOURCES[args.source](l, n=args.n) for l in args.langs}

    hdr = (f"{'lang':4} {'topk':>4} {'pools':>6} {'method':10} {'eta':>9} "
           f"{'d_eta%':>9} {'us/pool':>9} {'infeas':>7} {'lost':>6}")
    print(hdr)
    print("-" * len(hdr))

    with open(args.out, "w", encoding="utf-8") as fh:
        for lang in args.langs:
            for topk in args.topks:
                pools = collect_pools(lm, prompts[lang], topk, args)
                results = {n: evaluate(pools, n, s) for n, s in SOLVERS.items()}
                ref = results["exact"]["_per_pool"]
                for name, r in results.items():
                    mine = r["_per_pool"]
                    # A solver that gives up on its hardest instances must not be
                    # credited with the easy ones' average: compare both solvers
                    # only over the pools this one actually solved.
                    paired = [(a, b) for a, b in zip(mine, ref) if a is not None and b is not None]
                    lost = sum(1 for a, b in paired if a < b - 1e-12)
                    mine_mean = sum(a for a, _ in paired) / max(len(paired), 1)
                    ref_mean = sum(b for _, b in paired) / max(len(paired), 1)
                    d = 100 * (mine_mean - ref_mean) / ref_mean if ref_mean else 0.0

                    print(f"{lang:4} {topk:4d} {len(pools):6d} {name:10} "
                          f"{mine_mean:9.6f} {d:+9.5f} {r['us_per_pool']:9.1f} "
                          f"{r['infeasible']:7d} {lost:6d}")
                    rec = {k: v for k, v in r.items() if not k.startswith("_")}
                    rec.update({
                        "lang": lang, "topk": topk, "lost_vs_exact": lost,
                        # means restricted to the pools this solver solved, so the
                        # comparison is paired; `mean_eta` above is unrestricted.
                        "matched_pools": len(paired),
                        "matched_eta": mine_mean,
                        "matched_eta_exact": ref_mean,
                        "delta_eta_pct": d,
                    })
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
