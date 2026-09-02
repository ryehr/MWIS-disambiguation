"""End-to-end experiment driver.

For every (language, disambiguation method, top-k) cell it embeds a random
bitstream into generated text, extracts it back, and records the security and
capacity metrics.  Extraction is run every time -- an experiment that does not
verify the round trip is not measuring steganography.

  python scripts/run_experiments.py --langs en zh ja --methods exact greedy none \
      --topks 8 16 32 64 128 --n 50 --out runs/main.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch

from mwis_stego.antichain import EnumerationInfeasible
from mwis_stego.coder import CoderConfig, decode, encode
from mwis_stego.data import SOURCES
from mwis_stego.model import StegoLM


def run_cell(lm, prompts, method, topk, args, steps_fh=None):
    rows = []
    rng = random.Random(args.seed)
    for p in prompts:
        message = [rng.randint(0, 1) for _ in range(args.message_bits)]
        cfg = CoderConfig(precision=args.precision, topk=topk, temp=args.temp,
                          method=method, max_tokens=args.max_tokens, device=lm.device)
        ctx = lm.chat_context(p.text) if args.mode == "chat" else lm.raw_context(p.text)

        row = {"lang": p.lang, "key": p.key, "method": method, "topk": topk,
               "source": args.source, "mode": args.mode}
        try:
            t0 = time.perf_counter()
            ids, stego, es = encode(lm.model, lm.vocab, message, ctx, cfg, lm.banned_ids)
            t1 = time.perf_counter()
            if steps_fh is not None:
                for rec in es.steps_table():
                    rec.update({"lang": p.lang, "key": p.key, "method": method,
                                "topk": topk, "source": args.source})
                    steps_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            bits, _ = decode(lm.model, lm.vocab, stego, ctx, cfg, lm.banned_ids)
            t2 = time.perf_counter()

            # The encoder stops on whichever comes first: the message running out,
            # or the token budget.  Verify what was actually embedded, and report a
            # budget-limited run as `truncated` rather than as a decode failure.
            used = es.bits_used
            ok = bits[:used] == message[:used]
            truncated = used < len(message)
            s = es.summary()
            row.update(s)
            row.update({
                "status": "mismatch" if not ok else ("truncated" if truncated else "ok"),
                "tokens": len(ids),
                "bits": used,
                "bits_requested": len(message),
                "bpt": used / max(len(ids), 1),
                "chars": len(stego.decode("utf-8", errors="replace")),
                "encode_s": t1 - t0,
                "decode_s": t2 - t1,
                "text": stego.decode("utf-8", errors="replace"),
            })
        except EnumerationInfeasible as exc:
            row.update({"status": "infeasible", "component_size": exc.size})
        except Exception as exc:                              # noqa: BLE001
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        if args.verbose:
            print(f"    {row['status']:10} {p.key[:26]:26} "
                  f"bpt={row.get('bpt', 0):.3f} eta={row.get('mean_eta', 0):.4f}")
    return rows


def summarise(rows):
    ok = [r for r in rows if r["status"] in ("ok", "truncated")]
    if not ok:
        stat = {}
        for r in rows:
            stat[r["status"]] = stat.get(r["status"], 0) + 1
        return {"n": len(rows), "ok": 0, "status": stat}
    mean = lambda f: sum(r[f] for r in ok) / len(ok)
    return {
        "n": len(rows), "ok": len(ok),
        "bpt": mean("bpt"), "eta": mean("mean_eta"),
        "kld_c_bits": mean("kld_c_bits"), "ppl": mean("ppl"),
        "amb": mean("ambiguous_frac"), "kept": mean("mean_kept"),
        "encode_s": mean("encode_s"), "solve_s": mean("solve_seconds"),
        "mismatch": sum(1 for r in rows if r["status"] == "mismatch"),
        "truncated": sum(1 for r in rows if r["status"] == "truncated"),
        "infeasible": sum(1 for r in rows if r["status"] == "infeasible"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "ja"])
    ap.add_argument("--methods", nargs="+", default=["exact", "greedy", "enumerate", "none"])
    ap.add_argument("--topks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--source", default="flores", choices=list(SOURCES))
    ap.add_argument("--mode", default="chat", choices=["chat", "raw"])
    ap.add_argument("--n", type=int, default=50, help="prompts per cell")
    ap.add_argument("--message-bits", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--precision", type=int, default=26)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/main.jsonl")
    ap.add_argument("--steps-out", default="",
                    help="also write one row per generation step (per-step KL, eta, pool sizes)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lm = StegoLM(args.model, device=args.device)
    prompts = {l: SOURCES[args.source](l, n=args.n) for l in args.langs}

    hdr = f"{'lang':4} {'method':10} {'topk':>4} {'ok':>6} {'bpt':>6} {'eta':>7} " \
          f"{'kldc_b':>7} {'ppl':>8} {'amb':>5} {'kept':>5} {'enc_s':>6} {'solve_s':>8}"
    print(hdr)
    print("-" * len(hdr))

    steps_fh = None
    if args.steps_out:
        steps_path = Path(args.steps_out)
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        steps_fh = steps_path.open("w", encoding="utf-8")

    with out.open("w", encoding="utf-8") as fh:
        for lang in args.langs:
            for method in args.methods:
                for topk in args.topks:
                    rows = run_cell(lm, prompts[lang], method, topk, args, steps_fh)
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    s = summarise(rows)
                    if s["ok"]:
                        print(f"{lang:4} {method:10} {topk:4d} {s['ok']:3d}/{s['n']:<2d} "
                              f"{s['bpt']:6.3f} {s['eta']:7.4f} {s['kld_c_bits']:7.4f} "
                              f"{s['ppl']:8.2f} {s['amb']:5.2f} {s['kept']:5.1f} "
                              f"{s['encode_s']:6.2f} {s['solve_s']:8.4f}"
                              + (f"  MISMATCH={s['mismatch']}" if s["mismatch"] else "")
                              + (f"  trunc={s['truncated']}" if s["truncated"] else ""))
                    else:
                        print(f"{lang:4} {method:10} {topk:4d} {'0/' + str(s['n']):>6} "
                              f"  {s['status']}")
    if steps_fh is not None:
        steps_fh.close()
        print(f"wrote {args.steps_out}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
