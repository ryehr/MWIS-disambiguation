"""End-to-end: embed a random bitstream, then recover it from the stego text."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from mwis_stego.coder import CoderConfig, decode, encode
from mwis_stego.model import StegoLM

PROMPTS = {
    "en": "Write a short paragraph about the sea.",
    "zh": "写一段关于大海的短文。",
    "ja": "海について短い文章を書いてください。",
}


def main():
    lm = StegoLM(device="cuda")
    lm.vocab.verify([
        "running quickly", "他运行了一个程序。", "彼はプログラムを実行した。",
        "mixed 混合 テキスト 123", "café naïve ☕", "emoji 🌊 and 漢字",
    ])
    print("byte roundtrip over sample texts: OK")

    rng = random.Random(1234)
    message = [rng.randint(0, 1) for _ in range(64)]

    # `exact` and `greedy` must round-trip.  The other two are demonstrations,
    # not regressions: `none` is expected to fail, because that failure is the
    # evidence that segmentation ambiguity is real in each language, and
    # `enumerate` is expected to give up on English, because the infeasibility of
    # the paper's exponential baseline on byte-BPE pools is itself a finding.
    MUST_DECODE = {"exact", "greedy"}
    failures = 0
    for method in ["exact", "greedy", "enumerate", "none"]:
        for lang, prompt in PROMPTS.items():
            cfg = CoderConfig(method=method, topk=32, precision=26, max_tokens=200, device="cuda")
            ctx = lm.chat_context(prompt)
            try:
                ids, stego, es = encode(lm.model, lm.vocab, message, ctx, cfg, lm.banned_ids)
                got, _ = decode(lm.model, lm.vocab, stego, ctx, cfg, lm.banned_ids)
                ok = got[:len(message)] == message
            except Exception as exc:
                ids, stego, es, ok = [], b"", None, False
                err = f"{type(exc).__name__}: {exc}"
            expected = method in MUST_DECODE
            if es is None:
                verdict = "FAIL     " if expected else "as expected"
                failures += expected
                print(f"  {method:9} {lang}  {verdict}  {err}")
                continue
            s = es.summary()
            bpt = len(message) / max(len(ids), 1)
            text = stego.decode("utf-8", errors="replace")
            verdict = "OK      " if ok else ("MISMATCH" if expected else "undecodable")
            failures += expected and not ok
            print(f"  {method:9} {lang}  {verdict}  "
                  f"tokens={len(ids):3d} bpt={bpt:5.3f} eta={s['mean_eta']:.4f} "
                  f"kldc={s['kld_c_bits']:.4f}b ppl={s['ppl']:7.2f} amb={s['ambiguous_frac']:.2f}")
            print(f"             {text[:70]!r}")

    print("UNEXPECTED FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
