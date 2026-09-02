"""Steganalysis: can a detector separate disambiguated text from undisambiguated?

Both classes come from the same model, prompt, top-k cutoff, temperature and
token count (see `build_corpus.py`); the only difference is whether the candidate
pool was disambiguated. Detection accuracy is therefore a direct measure of the
distortion disambiguation introduces, and a solver that retains more probability
mass should be *harder* to detect. Chance is 50%.

Two detectors, corresponding to the paper's Accuracy-1 and Accuracy-2:

  ngram  character n-gram TF-IDF + logistic regression. Cheap, language-agnostic,
         no GPU, and a strong baseline for this kind of distributional shift.
  xlmr   XLM-RoBERTa fine-tuned as a binary classifier. One multilingual encoder
         across all three languages, so the numbers are comparable between them;
         per-language encoders would not be.

Splits are 60/20/20 train/val/test, stratified, and grouped by prompt key so that
the stego and cover texts written from the same prompt never straddle a split --
otherwise the detector can memorise the topic rather than learn the artefact.

  python scripts/steganalysis.py --corpus runs/corpus.jsonl --detector ngram
  python scripts/steganalysis.py --corpus runs/corpus.jsonl --detector xlmr --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(corpus):
    rows = [json.loads(l) for l in Path(corpus).open(encoding="utf-8")]
    cells = defaultdict(list)
    for r in rows:
        # One binary task per (lang, topk, stego method): that method vs `none`.
        if r["method"] == "none":
            continue
        cells[(r["lang"], r["topk"], r["method"])].append(r)
    covers = defaultdict(list)
    for r in rows:
        if r["method"] == "none":
            covers[(r["lang"], r["topk"])].append(r)
    return cells, covers


def split_by_key(rows, seed=0):
    """Group by prompt so a prompt's stego and cover texts land in the same split."""
    keys = sorted({r["key"] for r in rows})
    random.Random(seed).shuffle(keys)
    n = len(keys)
    tr, va = int(0.6 * n), int(0.8 * n)
    part = {k: ("train" if i < tr else "val" if i < va else "test")
            for i, k in enumerate(keys)}
    out = defaultdict(list)
    for r in rows:
        out[part[r["key"]]].append(r)
    return out["train"], out["val"], out["test"]


def run_ngram(train, val, test, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, max_features=200_000),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    clf.fit([r["text"] for r in train], [r["label"] for r in train])
    score = lambda rows: sum(
        int(p == r["label"]) for p, r in zip(clf.predict([r["text"] for r in rows]), rows)
    ) / max(len(rows), 1)
    return score(val), score(test)


def run_xlmr(train, val, test, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.detector_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.detector_model, num_labels=2
    ).to(args.device)

    def batches(rows, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            random.Random(args.seed).shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            chunk = [rows[j] for j in idx[i:i + args.batch_size]]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt").to(args.device)
            yield enc, torch.tensor([r["label"] for r in chunk], device=args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def accuracy(rows):
        model.eval()
        right = 0
        for enc, y in batches(rows, shuffle=False):
            right += int((model(**enc).logits.argmax(-1) == y).sum())
        return right / max(len(rows), 1)

    best_val, best_test = 0.0, 0.0
    for _ in range(args.epochs):
        model.train()
        for enc, y in batches(train, shuffle=True):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        v = accuracy(val)
        if v >= best_val:
            best_val, best_test = v, accuracy(test)
    return best_val, best_test


DETECTORS = {"ngram": run_ngram, "xlmr": run_xlmr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus.jsonl")
    ap.add_argument("--detector", default="ngram", choices=list(DETECTORS))
    ap.add_argument("--detector-model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cells, covers = load(args.corpus)
    results = []
    print(f"{'lang':4} {'topk':>4} {'method':10} {'n':>6} {'val':>7} {'test':>7}")
    print("-" * 46)
    for (lang, topk, method), stego in sorted(cells.items()):
        rows = stego + covers[(lang, topk)]
        if len({r["label"] for r in rows}) < 2:
            continue
        train, val, test = split_by_key(rows, seed=args.seed)
        v, t = DETECTORS[args.detector](train, val, test, args)
        print(f"{lang:4} {topk:4d} {method:10} {len(rows):6d} {v:7.4f} {t:7.4f}")
        results.append({"lang": lang, "topk": topk, "method": method,
                        "detector": args.detector, "n": len(rows),
                        "val_acc": v, "test_acc": t})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
