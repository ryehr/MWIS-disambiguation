# MWIS disambiguation for generative linguistic steganography

Reference implementation for *A Secure and Disambiguating Approach for Generative
Linguistic Steganography* (Yan, Yang and Song, IEEE Signal Processing Letters 30,
2023, pp. 1047–1051, [10.1109/LSP.2023.3302749](https://doi.org/10.1109/LSP.2023.3302749))
— and the corrections and extensions that came out of reworking it.

The original proof-of-concept, targeting CPM in Chinese, is preserved unmodified
under [`legacy/`](legacy/). The current implementation targets **Qwen3-0.6B** in
**English, Chinese and Japanese**.

> The name `legacy` is used twice, for two different things. The **`legacy/`
> directory** on this branch holds the three original source files, so they can
> be read beside the code that replaced them and line-referenced from §3 and §4.
> The **`legacy` branch** is the untouched snapshot of the repository as it stood
> when the letter was published — vocabulary files and all — for anyone who wants
> to reproduce the paper rather than this rework.

---

## 1. The problem

A generative steganography scheme selects the next token from the language
model's candidate pool, using the covert message as the selector. The receiver
replays the same model over the stego text and must recover the same token
sequence to invert the selection.

That inversion breaks when one candidate's bytes are a prefix of another's.
Given a pool containing both `大` and `大海`, a stego text beginning `大海…` is
explained by either, each carrying a different message. This is **segmentation
ambiguity**: the sender embedded one message, the receiver decodes several
candidates for it, and only one is right.

Formally, ambiguity occurs at a step whose pool `CP` admits
`W = {w₁, w₂, …} ⊆ CP`, `|W| > 1`, with every `w ∈ W` a prefix of the remaining
stego text `S`.

It must be eliminated *before* the pool is used to carve up the message space,
and identically on both sides.

## 2. Why maximum weight

Eliminating ambiguity means restricting the pool to a set with no internal prefix
relation — an independent set of the conflict graph

```
G = (V, E, w),   V = CP,   E = {(cᵢ, cⱼ) : cᵢ is a prefix of cⱼ, i ≠ j}
```

with node weights the token probabilities. Many independent sets exist; the
choice among them is exactly the security question, because renormalising the
retained pool `CP_a` distorts the model's distribution. With
`η = Σ_{i ∈ CP_a} pᵢ` the retained probability mass,

```
D_KL(CP_a ‖ CP) = Σ (pᵢ/η) · log((pᵢ/η)/pᵢ) = −log η
```

so **minimising the KL divergence is exactly maximising η** — a maximum weight
independent set (MWIS). This is the paper's central derivation and it stands.

## 3. Corrections to the published work

### 3.1 This MWIS instance is not NP-hard

The paper treats the problem as general MWIS (§III-C), cites its NP-hardness,
and therefore offers a choice between a greedy approximation and an exponential
enumeration. Neither is necessary.

"Is a prefix of" is a **partial order**, and the Hasse diagram of that order is a
**forest**: a candidate's parent is its longest proper prefix that is also a
candidate, and that is unique. Two candidates conflict exactly when one is an
ancestor of the other, so `G` is the *comparability graph of a forest poset*, and
an independent set in it is an **antichain**.

Maximum weight antichains in a forest are exactly solvable by a trie DP. Insert
every candidate's bytes into a trie; for a node `v` carrying weight `w(v)` (zero
if no candidate ends there):

```
f(v) = max( w(v),  Σ_{c ∈ children(v)} f(c) )
```

The two branches are the only options: `v` is comparable to everything in its
subtree, so taking `v` excludes all of it; otherwise the answer decomposes over
independent child subtrees. The optimum is `Σ f(root)`, computed in **O(Σ|cᵢ|)**
— linear in the total byte length of the pool.

So the exact solver is *faster* than the greedy heuristic (which spends O(k²)
just building the adjacency by pairwise `startswith`) while attaining the
enumeration baseline's optimum. See [`mwis_stego/antichain.py`](mwis_stego/antichain.py).
`greedy` and `enumerate_cc` are retained as baselines for comparison.

**The trie need not be materialised.** Pools are small — k ≤ 128 candidates of a
few bytes each — so in Python the constant factor decides, and allocating a node
object with a child dict per trie node costs far more than the traversal saves.
Sorting the candidates lexicographically exposes the same structure without any
of that: if `tᵢ` is a prefix of `tⱼ` then `i < j` in lex order, and every
candidate between them also starts with `tᵢ`. Two consequences:

- A pool contains a prefix conflict **iff some adjacent pair does**, so a single
  O(k) scan decides it. Between 14% and 57% of real pools are conflict-free
  depending on top-k, and those return immediately.
- Parent pointers of the prefix forest fall out of one stack pass, after which
  the DP is two flat array sweeps — no per-node object anywhere.

This is 9–11× faster than the explicit trie on real pools, and makes the
exact solver the fastest of the three at every pool size measured (§7.1). The trie
version is kept as `exact_trie`: it states the recurrence one-to-one and the
tests assert the two agree token for token, which is the property that matters —
both sides of the channel must retain the *identical* pool, not merely pools of
equal weight.

Verified in [`tests/test_antichain.py`](tests/test_antichain.py): over 3000
randomly generated prefix-structured pools, `exact` matches a brute-force MWIS
over all 2ⁿ subsets on every instance, as does `enumerate_cc`. Optimality is also
confirmed on real data — `exact` agrees with `enumerate_cc` on every pool the
latter can solve — 16 529 pools, no disagreement (§7.1).

**How much this is worth in η depends on the pool, and on real pools it is
small.** The synthetic pools above make `greedy` strictly suboptimal on 15.1% of
instances, but they are built from short random strings over a three-letter
alphabet and so are far denser in prefix relations than a real candidate pool;
that figure does not transfer. Measured on actual Qwen3 pools, `greedy` falls
short of the optimum on 0.1–25.6% of pools depending on top-k, and by a negligible
margin on average. The case for the exact solver rests on the other two axes: it
is optimal *by construction* rather than empirically close, and its cost grows
linearly rather than quadratically (`greedy`) or exponentially (`enumerate_cc`),
which makes it the fastest of the three in practice as well (§7.1).

### 3.2 The conditional skip in Algorithm 1 is not invertible

Algorithm 1 lines 1–2 skip disambiguation when the token originally intended for
output, `t_o`, has no prefix relation with the rest of the pool — presented as
the optimisation that keeps η high. **It cannot work**, and the published code
does not implement it (the attempt is commented out at
[`legacy/Embedding.py:320-342`](legacy/Embedding.py#L320-L342); `MWIS(...)` is
called unconditionally at [`legacy/Embedding.py:343`](legacy/Embedding.py#L343)).

The condition depends on `t_o`, which the sender derives from the *unmodified*
pool and which the receiver never observes. Take `CP_k = {B, A, AB}` where `B`
has no prefix relation with anything:

- **Skipped branch** — the interval lands on `B`; `B` has no prefix relation, so
  the full pool is kept and `B` is emitted under `CP_k`.
- **MWIS branch** — the interval lands on `A`, which conflicts with `AB`, so MWIS
  runs, yielding e.g. `CP_a = {B, A}`, arithmetic coding is redone over `CP_a`,
  and it may also emit `B`.

The receiver sees `B` and cannot tell which branch produced it. The interval
update divides by `η_k` in one case and `η_a` in the other, so the two paths
decode different messages. Extraction is ambiguous — the very failure the scheme
exists to prevent.

There is no repair. Any skip condition must be computable by the receiver, hence
a function of `CP_k` alone; the only such condition is "no pair in `CP_k`
conflicts", and on those pools MWIS already returns `CP_k` unchanged. **The skip
can only ever be a fast path, never a security gain.** The η improvement the
paper attributes to it does not exist.

Measured here, 39–90% of all generation steps contain a prefix conflict,
rising with top-k (§7.2), so the skip would rarely fire even if it were sound.

**Consequence for the algorithm's role:** since disambiguation now provably runs
at *every* step, its optimality and its cost stop being incidental. This is what
makes §3.1 the centre of the method rather than an implementation detail.

### 3.3 Prefix relations must be computed on bytes

The original compares decoded `str` values with `startswith`
([`legacy/Embedding.py:156`](legacy/Embedding.py#L156)). Under a byte-level BPE
vocabulary this is unsound: a single CJK character spans three UTF-8 bytes and
BPE routinely splits it across tokens. With Qwen3, `混` tokenises to the three
byte-fragment tokens `b' \xe6'`, `b'\xb7'`, `b'\xb7'`, none of which decodes to a
character — they render as replacement characters, and comparing those is
meaningless.

Every prefix test here operates on the exact bytes a token contributes to the
output stream. See [`mwis_stego/tokens.py`](mwis_stego/tokens.py); the byte map
is verified across all 151,669 Qwen3 vocabulary entries.

### 3.4 Ambiguity is not confined to unsegmented languages

The paper frames segmentation ambiguity as a property of unsegmented languages
(Chinese, Japanese) and treats English as unaffected. Under byte-level BPE that
is false: token boundaries do not align with word boundaries, so ` sea` and ` se`
coexist in the pool and are both prefixes of the same continuation.

Disabling disambiguation (`--methods none`) fails to round-trip in **all three
languages**, with representative conflicts:

| lang | conflicting candidates |
|---|---|
| en | `b' sea'` vs `b' se'`, `b' your'` vs `b' you'` |
| zh | `大海` vs `大` |
| ja | `前に` vs `前` |

The method therefore applies to any subword-tokenised model, not only to
unsegmented languages — which the paper's conclusion anticipated but did not
demonstrate.

## 4. Implementation changes

### 4.1 Arithmetic coder

Replaced with the reference implementation of Ziegler et al. 2019
([harvardnlp/NeuralSteganography](https://github.com/harvardnlp/NeuralSteganography),
reference [19] of the paper): integer interval arithmetic at fixed precision with
renormalisation. The original used `Decimal` at 400 digits of precision over a
`2**256` interval that never renormalises — workable but slow, and it compares
floats where the two sides must agree exactly. Integer counts remove that class
of desynchronisation entirely.

### 4.2 The decoder needs no BPE repair heuristic

Ziegler's decoder re-tokenises the stego text and matches by rank, then patches
up mis-tokenisation with a heuristic that can fail outright (printing
`Unable to fix BPE error`, [`arithmetic.py:195-237`](https://github.com/harvardnlp/NeuralSteganography/blob/master/arithmetic.py#L195-L237)).
That heuristic is a partial workaround for exactly the problem this method
solves.

Here the decoder walks the raw byte stream, and the antichain property makes the
match unique:

> If two retained tokens were both prefixes of the remaining stream, then — both
> being prefixes of the same string — the shorter would be a prefix of the
> longer, contradicting the antichain property. So **exactly one** retained token
> matches, and the sender's token sequence is recovered with no heuristic.

The decoder asserts this invariant rather than assuming it.

### 4.3 One pool constructor, shared by both sides

`candidate_pool()` in [`mwis_stego/coder.py`](mwis_stego/coder.py) is called
identically by `encode` and `decode`, so they cannot drift. The published version
duplicated ~250 lines verbatim between `Embedding.py` and `Extraction.py`
(`node`, `find_MWIS`, `BFS_Forest`, `DP`, `MWIS`, `find_connected_components`);
any divergence between the two copies is a silent decode failure.

### 4.4 The pool is enumerated with `topk`, not `multinomial`

The original built the pool with
`torch.multinomial(probs, num_samples=k)` ([`legacy/Embedding.py:280`](legacy/Embedding.py#L280)).
After top-k filtering only k tokens have non-zero probability, so sampling k
without replacement returns those k tokens **in a random permutation** — and that
order determines the arithmetic coder's interval assignment. Correctness relied
on the sender's and receiver's RNG streams staying in lockstep from a shared
seed, which any asymmetric early exit would silently break. (This is very likely
why §3.2's conditional skip had to be commented out: enabling it desynchronises
the two RNG streams.)

`torch.topk` is deterministic, ordered, faster, and uses no RNG.

### 4.5 Generation loop

KV cache and `torch.no_grad()`. The original re-ran a full forward pass over the
entire context at every step — O(L²) instead of O(L) — while building an autograd
graph it never used.

### 4.6 Disambiguation weights are integers

Probabilities handed to the solver are quantised to a fixed 32 bits, so which
tokens are retained is a pure function of the model's distribution, independent
of the coder's interval state, and every comparison inside the solver is exact.
A one-ULP float difference between the two sides can otherwise flip a `max` and
desynchronise extraction.

## 5. Setup

```bash
pip install -r requirements.txt
```

**Model.** `Qwen/Qwen3-0.6B` with thinking disabled — `enable_thinking=False`
pre-fills an empty `<think></think>` block so generation begins on the answer.
Weights load in **float32**: the two sides must obtain bit-identical logits, and
float16 on V100-class hardware does not reliably give that.

**Prompts.** Parallel FLORES passages — the same content professionally
translated into each language, so cross-language comparison is not confounded by
topic. FLORES is gated on the Hub; `facebook/belebele` is ungated and carries the
passages verbatim in its `flores_passage` field, aligned across languages by
`link`. **488 passages are parallel across English, Chinese and Japanese.** Each
is wrapped in a per-language instruction template so the model answers in that
language. `wikimedia/wikipedia` supplies unlimited non-parallel prompts for
building steganalysis corpora.

## 6. Running

```bash
# solver correctness: exact vs enumeration vs brute-force MWIS
python tests/test_antichain.py

# embed -> extract round trip, three languages, all methods
python tests/test_roundtrip.py

# solver quality on identical real pools
python scripts/compare_solvers.py --langs en zh ja --topks 8 16 32 64 128 --n 12

# end-to-end security and capacity metrics
python scripts/run_experiments.py --langs en zh ja \
    --methods exact greedy enumerate none --topks 8 32 128 --n 12 \
    --max-tokens 512 --out runs/main.jsonl
```

`compare_solvers.py` records the pools an actual generation run encounters and
replays every solver on the **same** pools. Running each method end to end
instead lets the generated texts diverge, which mixes solver quality with the
luck of the sampled text.

`run_experiments.py` verifies extraction on **every** sample — an experiment that
does not check the round trip is not measuring steganography. Statuses are
distinguished: `ok`, `truncated` (the token budget ran out before the message
did; the embedded prefix still verifies), `mismatch` (a genuine decode failure),
`infeasible` (the enumeration baseline hit a component it cannot enumerate).

### Metrics

Two distortions are stacked in this pipeline and the metrics keep them apart.
The top-k cutoff discards probability mass *before* disambiguation runs, and it
is applied identically by every method — `none` included. Reporting η against the
whole vocabulary would therefore charge disambiguation for the truncation, and
`none` would show a non-zero KLD-c despite removing nothing. All pool-relative
quantities below are measured against the top-k pool, so `none` sits at exactly
η = 1 and its residual KLD-c is the arithmetic coder's integer quantisation
floor — which every method pays and which should be subtracted before comparing.

| name | meaning |
|---|---|
| `eta` | `η_a`, retained mass **relative to the top-k pool**; `−log η_a` is the KL |
| `eta_vocab` | retained mass relative to the whole vocabulary (carries the cutoff too) |
| `pool_mass` | mass the top-k cutoff keeps; the two above differ by this factor |
| `kldc_b` | `D_KL(CP_a ‖ CP)` in bits, exact on the quantised pool, pool-relative |

`--steps-out` on `run_experiments.py` writes one row per generation step rather
than one per text — per-step KL, η, pool and retained sizes, and whether the pool
conflicted. Per-step KL is strongly right-skewed (§7.3), so a mean over steps is
a poor summary of it and the raw series is worth keeping.
| `bpt` | bits embedded per generated token |
| `ppl` | perplexity of the stego text under the unmodified model |
| `amb` | fraction of steps whose pool contained a prefix conflict |
| `solve_s` | wall time inside the disambiguation solver, per stego text |

## 7. Results

### 7.1 Solvers, on identical pools

All three solvers replayed on the same recorded pools — **17 087 pools** drawn
from real generation runs across the three languages and five top-k settings
(`scripts/compare_solvers.py`, plus `scripts/bench_solvers.py` for timings).
Because every solver sees the identical pools, a difference is the solver's and
not the sampled text's.

| lang | top-k | pools | `exact` µs | `greedy` µs | `enumerate` µs | `greedy` slower | `enumerate` slower | `greedy` loses on | `enumerate` infeasible |
|---|---|---|---|---|---|---|---|---|---|
| en | 8 | 1554 | **5.0** | 24.3 | 27.7 | 4.8× | 5.5× | 5 (0.3%) | 0 |
| en | 16 | 1872 | **6.9** | 46.8 | 2 606 | 6.7× | 376× | 10 (0.5%) | 0 |
| en | 32 | 963 | **20.1** | 253.5 | 82 082 | 12.6× | 4 092× | 27 (2.8%) | 41 (4.3%) |
| en | 64 | 934 | **38.8** | 865.1 | 87 181 | 22.3× | 2 247× | 89 (9.5%) | 101 (10.8%) |
| en | 128 | 907 | **75.9** | 2 937 | 285 850 | 38.7× | 3 764× | 232 (25.6%) | 222 (24.5%) |
| zh | 8 | 1083 | **6.3** | 29.5 | 33.5 | 4.7× | 5.3× | 1 (0.1%) | 0 |
| zh | 16 | 892 | **11.7** | 82.9 | 995.9 | 7.1× | 85× | 11 (1.2%) | 0 |
| zh | 32 | 911 | **21.0** | 256.3 | 117 626 | 12.2× | 5 611× | 23 (2.5%) | 10 (1.1%) |
| zh | 64 | 907 | **40.1** | 885.9 | 117 759 | 22.1× | 2 938× | 40 (4.4%) | 22 (2.4%) |
| zh | 128 | 824 | **77.9** | 3 117 | 372 775 | 40.0× | 4 788× | 53 (6.4%) | 60 (7.3%) |
| ja | 8 | 1431 | **6.5** | 30.8 | 35.8 | 4.7× | 5.5× | 10 (0.7%) | 0 |
| ja | 16 | 1276 | **12.3** | 86.2 | 160.1 | 7.0× | 13× | 19 (1.5%) | 0 |
| ja | 32 | 1228 | **21.2** | 258.2 | 46 624 | 12.2× | 2 202× | 40 (3.3%) | 0 |
| ja | 64 | 1115 | **43.3** | 952.0 | 110 385 | 22.0× | 2 549× | 105 (9.4%) | 17 (1.5%) |
| ja | 128 | 1190 | **83.3** | 3 256 | 431 293 | 39.1× | 5 178× | 195 (16.4%) | 85 (7.1%) |

**Optimality holds on real data.** `enumerate_cc` searches every subset of every
connected component, so where it runs it is optimal by definition. It agrees with
`exact` on **every one of the 16 529 pools it could solve** — all fifteen cells,
no disagreement in weight and none in membership. The forest-antichain argument
of §3.1 is not only a proof but a checked one.

**`exact` is faster than `greedy` everywhere, and the gap widens with top-k** —
4.7× at top-8 rising to 40× at top-128, consistently across all three languages.
The measured scaling is what the analysis predicts: over a 16× increase in pool
size `exact` grows 12–15× (linear) while `greedy` grows 106–121× (quadratic),
its O(k²) adjacency construction taking over well before top-128.

**The enumeration baseline is infeasible, not merely slow.** It is 5× to 5 600×
slower than `exact`, and it fails outright on **558 of 17 087 pools (3.3%)** —
concentrated where pools are large, reaching **24.5% of English pools at
top-128**, where a single connected component can span most of the pool. The
paper's claim that enumeration "may still be a fast way in small-scale connected
components" reflects testing only Chinese CPM.

**`greedy`'s suboptimality is rare and cheap at small top-k, and neither at
large.** It loses on 0.1–0.7% of pools at top-8 but on 6.4–25.6% at top-128. Its
*aggregate* η cost stays tiny throughout — between −0.00001% and −0.015% — so the
case for `exact` is not that it buys much η on average. It is that `exact` is
optimal by construction rather than usually-close, and is the cheaper of the two
at every size measured.

> A caveat worth carrying into any write-up: an *unrestricted* mean η flatters
> `enumerate_cc`, because it averages only over the pools it could solve and
> silently drops the hardest ones. `compare_solvers.py` therefore reports means
> restricted to each solver's own solved set alongside `exact` on that same
> subset. Compared that way, `enumerate_cc` never beats `exact` on any pool.

### 7.2 End to end

Full matrix: 3 languages × 4 methods × top-k ∈ {8, 32, 128}, 12 parallel FLORES
prompts per cell, 64-bit messages, 512-token budget. **Extraction is run on every
sample.**

| lang | method | top-k | ok | bpt | η_a | KLD-c (bits) | PPL | amb | kept | solver ms |
|---|---|---|---|---|---|---|---|---|---|---|
| en | `exact` | 8 | 12/12 | 0.661 | 0.9875 | 0.0250 | 2.09 | 0.39 | 6.5 | **1.7** |
| en | `exact` | 32 | 12/12 | 0.836 | 0.9837 | 0.0306 | 2.41 | 0.78 | 23.3 | **2.6** |
| en | `exact` | 128 | 12/12 | 0.913 | 0.9818 | 0.0317 | 2.62 | 0.87 | 76.2 | **6.7** |
| en | `greedy` | 8 | 12/12 | 0.652 | 0.9880 | 0.0237 | 2.08 | 0.37 | 6.5 | 5.5 |
| en | `greedy` | 32 | 12/12 | 0.861 | 0.9849 | 0.0274 | 2.47 | 0.78 | 23.1 | 23.5 |
| en | `greedy` | 128 | 12/12 | 0.865 | 0.9833 | 0.0283 | 2.47 | 0.85 | 74.2 | 236.1 |
| en | `enumerate` | 8 | 12/12 | 0.661 | 0.9875 | 0.0250 | 2.09 | 0.39 | 6.5 | 5.6 |
| en | `enumerate` | 32 | 4/12 | — | — | — | — | — | — | 859.2 |
| en | `enumerate` | 128 | 0/12 | \* infeasible | | | | | | |
| en | `none` | any | 0/12 | \* undecodable | | | | | | |
| zh | `exact` | 8 | 12/12 | 0.742 | 0.9660 | 0.0578 | 2.33 | 0.66 | 6.2 | **1.5** |
| zh | `exact` | 32 | 12/12 | 0.899 | 0.9617 | 0.0636 | 2.71 | 0.87 | 21.4 | **2.6** |
| zh | `exact` | 128 | 12/12 | 1.096 | 0.9601 | 0.0673 | 3.29 | 0.88 | 75.8 | **6.2** |
| zh | `greedy` | 8 | 12/12 | 0.737 | 0.9662 | 0.0578 | 2.33 | 0.66 | 6.2 | 4.4 |
| zh | `greedy` | 32 | 12/12 | 0.872 | 0.9646 | 0.0596 | 2.62 | 0.86 | 21.4 | 24.9 |
| zh | `greedy` | 128 | 12/12 | 1.032 | 0.9577 | 0.0701 | 3.03 | 0.88 | 72.6 | 224.0 |
| zh | `enumerate` | 32 | 7/12 | — | — | — | — | — | — | 9 614.4 |
| ja | `exact` | 8 | 12/12 | 0.607 | 0.9674 | 0.0574 | 2.14 | 0.63 | 6.3 | **2.1** |
| ja | `exact` | 32 | 12/12 | 0.694 | 0.9630 | 0.0627 | 2.20 | 0.85 | 20.6 | **3.5** |
| ja | `exact` | 128 | 12/12 | 0.736 | 0.9554 | 0.0744 | 2.28 | 0.90 | 73.9 | **9.6** |
| ja | `greedy` | 8 | 12/12 | 0.607 | 0.9684 | 0.0557 | 2.11 | 0.64 | 6.3 | 5.9 |
| ja | `greedy` | 32 | 12/12 | 0.718 | 0.9624 | 0.0626 | 2.26 | 0.87 | 20.9 | 30.7 |
| ja | `greedy` | 128 | 12/12 | 0.884 | 0.9628 | 0.0625 | 2.76 | 0.90 | 75.9 | 296.4 |
| ja | `enumerate` | 32 | 12/12 | 0.694 | 0.9630 | 0.0627 | 2.20 | 0.85 | 20.6 | 4 805.6 |

**Extraction is exact: zero decode mismatches** over all 275 samples that
produced text, across every language, method and top-k. Three samples are
`truncated` — the 512-token budget bound before the 64-bit message did — and the
bits they did embed still verify.

**`none` fails in all 108 attempts**, in all three languages and at every top-k:
without disambiguation there is no decodable channel at all (§3.4).

**`enumerate_cc` cannot complete the matrix.** It is infeasible on all 36
top-128 samples, and on 8 of 12 English and 5 of 12 Chinese samples at top-32.
Its surviving cells are struck from the table because they are a *biased* subset
— the samples whose pools happened to stay small — and averaging them would
reward the baseline for the instances it failed to solve.

**Solver cost tracks §7.1**: at top-128 `exact` spends 6–10 ms per stego text
against `greedy`'s 224–296 ms (~30×) and `enumerate_cc`'s seconds.

**Ambiguity is pervasive and grows with top-k**: 0.39 → 0.87 of steps in English,
0.66 → 0.88 in Chinese, 0.63 → 0.90 in Japanese, between top-8 and top-128.

> **Do not read solver quality out of this table.** Each method generates its own
> text, so the methods walk different trajectories and meet different pools; the
> per-method η and KLD-c columns differ for that reason as much as for any
> property of the solver. At n = 12 the trajectory variance dominates outright —
> which is why `greedy` posts a *lower* KLD-c than `exact` in several cells here
> even though §7.1 shows `exact` weakly better on **every individual pool**.
> Solver comparisons belong on identical pools; that is what §7.1 is for, and
> what `compare_solvers.py` exists to measure.


### 7.3 Per-step KL

`--steps-out` writes one record per generation step. Aggregating those into a
mean, as the paper's KLD-c does, throws away the shape of the distribution — and
the shape is the finding. **20 229 steps**, `exact` and `greedy` pooled:

| lang | top-k | steps | conflicting | mean | median | p75 | p90 | p99 | max | top 1% of steps carry | top 10% carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 8 | 3134 | 36.6% | 0.0181 | **0.00000** | 0.0017 | 0.0372 | 0.4055 | 0.688 | 29.1% | 89.3% |
| en | 32 | 1917 | 78.5% | 0.0285 | 0.00109 | 0.0119 | 0.0740 | 0.4590 | 1.028 | 23.5% | 78.7% |
| en | 128 | 1868 | 86.3% | 0.0289 | 0.00151 | 0.0141 | 0.0723 | 0.4816 | 0.906 | 22.9% | 76.2% |
| zh | 8 | 2176 | 66.4% | 0.0555 | 0.00456 | 0.0556 | 0.1864 | 0.5051 | 1.258 | 12.6% | 59.7% |
| zh | 32 | 1859 | 86.6% | 0.0588 | 0.01023 | 0.0568 | 0.1735 | 0.5918 | 0.979 | 12.1% | 59.2% |
| zh | 128 | 1642 | 87.5% | 0.0644 | 0.01375 | 0.0709 | 0.1973 | 0.5569 | 0.807 | 10.5% | 55.3% |
| ja | 8 | 2868 | 64.1% | 0.0526 | 0.00178 | 0.0352 | 0.1665 | 0.6185 | 1.136 | 14.6% | 70.8% |
| ja | 32 | 2505 | 84.0% | 0.0553 | 0.00460 | 0.0443 | 0.1713 | 0.6707 | 1.322 | 15.2% | 65.9% |
| ja | 128 | 2260 | 89.2% | 0.0634 | 0.00608 | 0.0539 | 0.1852 | 0.6814 | 1.316 | 13.7% | 64.6% |

Bits per step; the distribution is over steps, not over texts.

**The cost is concentrated in a few steps, and the mean hides that.** Across all
20 229 steps the **top 1% carry 15.8% of the total KL and the top 10% carry
69.4%**. In English at top-8 the median step costs *nothing at all* while p99 is
0.41 bits — the p99 exceeds the mean by 22× and the p90 by 11×. At top-32 the p99
is over 400× the median. The mean KLD-c that the paper reports is an average over
a distribution most of whose mass sits at zero.

**Most steps are free even where conflicts are common.** 8.8% of steps have
numerically zero KL, and at top-8 in English 63% of steps have no prefix conflict
at all, so disambiguation is a no-op on them. Conflicts become near-universal as
top-k grows (36.6% → 86.3% in English), but the median cost stays small: what
grows is the tail.

**Which language is expensive is not what §3.4 might suggest.** English carries
roughly half the per-step KL of Chinese or Japanese — its pools conflict less
often and less severely — but its *tail* is the most concentrated (top 10% of
steps carry 89.3% of the cost at top-8, against 55–60% for Chinese). So English
is cheaper on average and lumpier, which matters for any detector that looks at
per-step statistics rather than aggregates.

This also says where a better scheme would pay off: not in shaving the median,
which is already zero, but in the handful of steps where a heavy candidate has to
be dropped.

## 8. Layout

```
mwis_stego/antichain.py   exact / greedy / enumeration solvers + brute-force reference
mwis_stego/tokens.py      byte-level view of the vocabulary
mwis_stego/coder.py       arithmetic coder + disambiguation; encode and decode
mwis_stego/model.py       Qwen3 loading, chat and raw contexts
mwis_stego/data.py        parallel FLORES and Wikipedia prompts
scripts/run_experiments.py    end-to-end metrics, round trip verified per sample
scripts/compare_solvers.py    solver quality on identical recorded pools
scripts/bench_solvers.py      solver cost on identical recorded pools
scripts/build_corpus.py       paired stego / cover corpora for steganalysis
scripts/steganalysis.py       n-gram and XLM-R detectors
tests/test_antichain.py       solvers vs brute-force MWIS; exact vs exact_trie
tests/test_roundtrip.py       embed -> extract, three languages
legacy/                       the original CPM implementation, unmodified
```

## 9. Citation

```bibtex
@ARTICLE{10215094,
  author={Yan, Ruiyi and Yang, Yating and Song, Tian},
  journal={IEEE Signal Processing Letters}, 
  title={A Secure and Disambiguating Approach for Generative Linguistic Steganography}, 
  year={2023},
  volume={30},
  number={},
  pages={1047-1051},
  keywords={Security;Codes;Steganography;Receivers;Ice;Probability distribution;Linguistics;Linguistic steganography;maximum weight independent set;segmentation ambiguity;disambiguation},
  doi={10.1109/LSP.2023.3302749}}
```

The arithmetic coder derives from
[harvardnlp/NeuralSteganography](https://github.com/harvardnlp/NeuralSteganography)
(Ziegler, Deng and Rush, *Neural Linguistic Steganography*, EMNLP-IJCNLP 2019).
