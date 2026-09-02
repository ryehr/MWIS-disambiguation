# MWIS disambiguation for generative linguistic steganography

Reference implementation for *A Secure and Disambiguating Approach for Generative
Linguistic Steganography* (Yan, Yang and Song, IEEE Signal Processing Letters 30,
2023, pp. 1047–1051, [10.1109/LSP.2023.3302749](https://doi.org/10.1109/LSP.2023.3302749))
— and the corrections and extensions that came out of reworking it.

The original proof-of-concept, targeting CPM in Chinese, is preserved unmodified
under [`legacy/`](legacy/). The current implementation targets **Qwen3-0.6B** in
**English, Chinese and Japanese**.

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

This is 9–11× faster than the explicit trie on real pools (§7.1). The trie
version is kept as `exact_trie`: it states the recurrence one-to-one and the
tests assert the two agree token for token, which is the property that matters —
both sides of the channel must retain the *identical* pool, not merely pools of
equal weight.

Verified in [`tests/test_antichain.py`](tests/test_antichain.py): over 3000
randomly generated prefix-structured pools, `exact` matches a brute-force MWIS
over all 2ⁿ subsets on every instance, as does `enumerate_cc`. Optimality is also
confirmed on real data — `exact` agrees with `enumerate_cc` on every pool the
latter can solve (§7.2).

**How much this is worth in η depends on the pool, and on real pools it is
small.** The synthetic pools above make `greedy` strictly suboptimal on 15.1% of
instances, but they are built from short random strings over a three-letter
alphabet and so are far denser in prefix relations than a real candidate pool;
that figure does not transfer. Measured on actual Qwen3 pools, `greedy` falls
short of the optimum on 0.3–2.8% of pools depending on top-k, and by a negligible
margin on average. The case for the exact solver rests on the other two axes: it
is optimal *by construction* rather than empirically close, and its cost grows
linearly rather than quadratically (`greedy`) or exponentially (`enumerate_cc`).

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

Measured here, 64–89% of all generation steps contain a prefix conflict
(§6), so the skip would rarely fire even if it were sound.

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

| name | meaning |
|---|---|
| `eta` | retained probability mass `η`; the quantity `−log η` penalises |
| `kldc_b` | `D_KL(CP_a ‖ CP)` in bits, computed exactly on the quantised pool |
| `bpt` | bits embedded per generated token |
| `ppl` | perplexity of the stego text under the unmodified model |
| `amb` | fraction of steps whose pool contained a prefix conflict |
| `solve_s` | wall time inside the disambiguation solver, per stego text |

## 7. Results

### 7.1 Solver cost

Microseconds per pool, on pools recorded from real English generation runs
(`scripts/bench_solvers.py`, best of 5 passes):

| top-k | pools | with a conflict | `exact` | `exact_trie` | `greedy` |
|---|---|---|---|---|---|
| 8 | 531 | 42.9% | **4.5** | 49.9 | 29.5 |
| 16 | 687 | 63.3% | **7.4** | 71.9 | 61.5 |
| 32 | 458 | 79.0% | **18.8** | 179.1 | 256.7 |
| 64 | 475 | 84.8% | **36.8** | 343.1 | 854.7 |
| 128 | 447 | 86.1% | **71.4** | 660.6 | 2 985.5 |

- **`exact` is faster than `greedy` at every pool size** — 6.6× at top-8, rising
  to **41.8× at top-128**.
- **The measured scaling matches the analysis.** Over a 16× increase in pool size
  `exact` grows 15.9× (linear) while `greedy` grows 101× (quadratic): its O(k²)
  adjacency construction dominates well before top-128.
- Dropping the explicit trie is worth **9–11×** across the whole range (§3.1).

### 7.2 Solver quality, on identical pools

Every solver replayed on the same recorded English pools (Qwen3-0.6B, chat mode,
12 FLORES prompts per cell). `exact` and `greedy` solve every pool, so their
means are directly paired; `enumerate_cc` gives up on the largest components.
(Timings here are from an earlier build and are superseded by §7.1; what this
table establishes is *quality* and *feasibility*.)

| top-k | pools | `exact` η | `greedy` Δη | `greedy` loses on | `enumerate` infeasible | `enumerate` µs/pool |
|---|---|---|---|---|---|---|
| 8 | 1554 | 0.990457 | −2.7 × 10⁻⁶ % | 5 (0.3%) | 0 | 29.5 |
| 16 | 1872 | 0.990525 | −9.5 × 10⁻⁶ % | 10 (0.5%) | 0 | 2 638 |
| 32 | 963 | 0.983814 | −1.4 × 10⁻³ % | 27 (2.8%) | 41 (4.3%) | 82 481 |

- **`exact` agrees with `enumerate_cc` on every pool the latter solves**, which
  confirms optimality on real data and not only on the synthetic instances of
  §3.1.
- **`greedy`'s shortfall is real but small, and grows with top-k** — from 0.3% to
  2.8% of pools between top-8 and top-32. Its *average* η cost is negligible. The
  argument for `exact` is that it is optimal by construction, and cheaper (§7.1).
- **The enumeration baseline is not merely slow — it is infeasible.** Its cost
  rises 2 800× from top-8 to top-32, and at top-32 it hits connected components
  of up to **31 nodes** (2³¹ subsets) on 4.3% of pools. The paper's claim that
  enumeration "may still be a fast way in small-scale connected components"
  reflects testing only Chinese CPM; English byte-level BPE pools can be almost
  entirely one component.

> A caveat worth carrying into any write-up: `enumerate_cc`'s *unrestricted* mean
> η looks **higher** than `exact`'s at top-32, because it averages only over the
> pools it could solve and silently drops the 41 hardest. `compare_solvers.py`
> therefore reports means restricted to each solver's own solved set alongside
> `exact` on that same subset. Compared properly, `enumerate_cc` never beats
> `exact` on any pool.

### 7.3 End to end

The full matrix (3 languages × 4 methods × top-k ∈ {8, 32, 128}) is produced by
`scripts/run_experiments.py`; see `runs/`. Extraction is verified on every
sample.

Established so far:

- **`none` never round-trips**, in any of the three languages — see §3.4.
- **Round-trip integrity holds** for `exact` and `greedy`: zero decode mismatches
  across all samples run to date.
- **Solver cost is negligible end to end**: 7–14 ms per stego text of 60–95
  tokens, roughly 0.1 ms per step, against one full model forward pass per step.
  Whatever the paper's Table II is measuring, it is not dominated by the solver.

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
@article{yan2023secure,
  title   = {A Secure and Disambiguating Approach for Generative Linguistic Steganography},
  author  = {Yan, Ruiyi and Yang, Yating and Song, Tian},
  journal = {IEEE Signal Processing Letters},
  volume  = {30},
  pages   = {1047--1051},
  year    = {2023},
  doi     = {10.1109/LSP.2023.3302749}
}
```

The arithmetic coder derives from
[harvardnlp/NeuralSteganography](https://github.com/harvardnlp/NeuralSteganography)
(Ziegler, Deng and Rush, *Neural Linguistic Steganography*, EMNLP-IJCNLP 2019).
