"""Arithmetic-coding linguistic steganography with prefix disambiguation.

The arithmetic coder is the reference implementation of Ziegler et al. 2019
(harvardnlp/NeuralSteganography): the message is a binary fraction, the current
interval is kept as two `precision`-bit integers, and per step the candidate
probabilities are quantised into integer counts spanning that interval.  Integer
counts are what make the scheme reproducible -- the sender and the receiver
compare exact integers, never floats.

Two changes to the reference:

1. `candidate_pool` inserts disambiguation between building the pool and
   quantising it.  The retained pool is an antichain of the prefix order, so no
   retained token is a prefix of another.
2. `decode` walks the stego *byte stream* instead of re-tokenising the text and
   matching by rank.  The reference has to repair mis-tokenisation at decode
   time with a heuristic that can fail outright ("Unable to fix BPE error");
   after disambiguation there is nothing to repair, because:

       If two retained tokens were both prefixes of the remaining stream, the
       shorter would be a prefix of the longer, contradicting the antichain
       property.  So exactly one retained token matches, and the receiver
       recovers the sender's token sequence with no ambiguity and no heuristic.

Both sides derive the pool through the *same* function, so they cannot drift.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .antichain import SOLVERS

__all__ = ["CoderConfig", "encode", "decode", "StepStats"]

# Weights handed to the disambiguation solver are probabilities quantised to
# this many bits.  Fixed, so that which tokens are retained depends only on the
# model's distribution -- not on the coder's interval state.
WEIGHT_PRECISION = 32


def bits2int(bits) -> int:
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp: int, num_bits: int) -> list[int]:
    if num_bits == 0:
        return []
    return [int(c) for c in reversed(("{0:0%db}" % num_bits).format(inp))]


def num_same_from_beg(bits1, bits2) -> int:
    """Length of the common prefix of two equal-length bit lists.

    The reference returns len-1 when the lists are identical; returning the true
    length instead consumes the extra determined bit.  Encode and decode share
    this function, so the change stays symmetric.
    """
    assert len(bits1) == len(bits2)
    for i, (a, b) in enumerate(zip(bits1, bits2)):
        if a != b:
            return i
    return len(bits1)


@dataclass
class CoderConfig:
    precision: int = 26
    topk: int = 64                  # hard cap on pool size, as in the paper's top-k
    temp: float = 1.0
    method: str = "exact"           # exact | greedy | enumerate | none
    max_tokens: int = 512
    device: str = "cuda"


@dataclass
class StepStats:
    """Per-step record; the security metrics of the paper are aggregates of these."""
    steps: int = 0
    pool_sizes: list[int] = field(default_factory=list)
    kept_sizes: list[int] = field(default_factory=list)
    etas: list[float] = field(default_factory=list)        # retained mass vs the whole vocabulary
    pool_masses: list[float] = field(default_factory=list)  # mass the top-k cutoff keeps
    etas_pool: list[float] = field(default_factory=list)   # eta_a: retained mass vs the pool
    kld_c: list[float] = field(default_factory=list)       # -log(eta_a), nats
    kld_c_true: list[float] = field(default_factory=list)  # exact KL(q || p_pool) in bits
    logprobs: list[float] = field(default_factory=list)    # under the *model*, for perplexity
    ambiguous_steps: int = 0                               # pools that had a prefix conflict
    bits_used: int = 0                                     # message bits actually embedded
    solve_seconds: float = 0.0                             # time inside the disambiguation solver
    record_pools: bool = False                             # keep raw pools for offline solver comparison
    pools: list = field(default_factory=list)              # (token_bytes, integer weights)

    def summary(self) -> dict:
        n = max(self.steps, 1)
        return {
            "steps": self.steps,
            "mean_pool": sum(self.pool_sizes) / n,
            "mean_kept": sum(self.kept_sizes) / n,
            "mean_eta": sum(self.etas_pool) / n,       # eta_a, pool-relative
            "mean_eta_vocab": sum(self.etas) / n,      # includes the top-k cutoff
            "mean_pool_mass": sum(self.pool_masses) / n,
            "kld_c_nats": sum(self.kld_c) / n,
            "kld_c_bits": sum(self.kld_c_true) / n,
            "ppl": math.exp(-sum(self.logprobs) / n),
            "ambiguous_frac": self.ambiguous_steps / n,
            "solve_seconds": self.solve_seconds,
        }


def candidate_pool(logits_row, cur_interval, vocab, cfg, banned_ids, stats=None):
    """Build the retained, quantised candidate pool for one generation step.

    Called identically by the sender and the receiver.  Returns
    ``(token_ids, token_bytes, cum_probs)`` where ``cum_probs`` are the integer
    interval boundaries and ``token_ids[j]`` owns ``[cum_probs[j-1], cum_probs[j])``.
    """
    logits_row = logits_row.clone()
    logits_row[banned_ids] = -1e20

    logits, indices = logits_row.sort(descending=True, stable=True)
    logits = logits.double() / cfg.temp
    probs = F.softmax(logits, dim=0)

    cur_int_range = cur_interval[1] - cur_interval[0]
    cur_threshold = 1.0 / cur_int_range
    below = (probs < cur_threshold).nonzero()
    dyn_k = below[0].item() if len(below) else probs.numel()
    k = min(max(2, dyn_k), cfg.topk)

    pool_ids = indices[:k].tolist()
    pool_probs = probs[:k]
    pool_bytes = [vocab[i] for i in pool_ids]

    # An empty byte string is a prefix of every other string, so a single one in
    # the pool makes every candidate conflict with it and the antichain collapses
    # to one token -- without raising anything.  Ids that decode to no bytes are
    # banned in `StegoLM.banned_ids`; this catches any that get past that.
    if not all(pool_bytes):
        empty = [i for i, b in zip(pool_ids, pool_bytes) if not b]
        raise AssertionError(f"candidate pool contains ids with no bytes: {empty}")

    # --- disambiguation -------------------------------------------------
    weights = (pool_probs * (1 << WEIGHT_PRECISION)).round().long().tolist()
    if cfg.method == "none":
        kept = list(range(k))
    else:
        t0 = time.perf_counter()
        kept = SOLVERS[cfg.method](pool_bytes, weights)
        if stats is not None:
            stats.solve_seconds += time.perf_counter() - t0
    if not kept:                       # every weight rounded to zero
        kept = [0]

    # Two distortions are stacked here and must not be conflated.  The top-k
    # cutoff already discards mass before disambiguation runs, and it is applied
    # identically by every method -- including `none`.  Metrics measured against
    # the whole vocabulary carry both; the paper's eta_a is the pool-relative
    # one, which isolates disambiguation and is exactly 1 (KLD-c exactly 0) when
    # nothing is removed.
    pool_mass = float(pool_probs.sum())

    if stats is not None and stats.record_pools:
        stats.pools.append((pool_bytes, weights))
    if stats is not None:
        conflict = len(kept) < k
        stats.pool_sizes.append(k)
        stats.kept_sizes.append(len(kept))
        eta = float(pool_probs[kept].sum())
        eta_pool = eta / pool_mass if pool_mass > 0 else 0.0
        stats.etas.append(eta)
        stats.pool_masses.append(pool_mass)
        stats.etas_pool.append(eta_pool)
        stats.kld_c.append(-math.log(eta_pool) if eta_pool > 0 else 0.0)
        if conflict:
            stats.ambiguous_steps += 1

    kept_t = torch.tensor(kept, device=probs.device, dtype=torch.long)
    kept_probs = probs[kept_t]

    # --- quantise into the current interval (reference procedure) --------
    q = kept_probs / kept_probs.sum() * cur_int_range
    q = q.round().long()
    cum = q.cumsum(0)

    overfill = (cum > cur_int_range).nonzero()
    if len(overfill) > 0:
        cut = max(int(overfill[0].item()), 1)
        cum = cum[:cut]
        kept = kept[:cut]
        kept_t = kept_t[:cut]
    cum = cum + (cur_int_range - cum[-1])

    if stats is not None:
        final = cum.clone()
        final[1:] = cum[1:] - cum[:-1]
        qq = final.double() / final.sum()
        logq = qq.log()
        # p renormalised over the pool, for the same reason as eta_pool above:
        # this is the cost of disambiguating, not of the top-k cutoff that every
        # method shares.  It is 0 when nothing is removed, up to quantisation.
        logp = F.log_softmax(logits, dim=0)[kept_t] - math.log(max(pool_mass, 1e-300))
        contrib = qq * (logq - logp) / 0.69314718055994531
        contrib[qq == 0] = 0
        stats.kld_c_true.append(float(contrib.sum()))

    cum = cum + cur_interval[0]
    token_ids = [pool_ids[j] for j in kept]
    token_bytes = [pool_bytes[j] for j in kept]
    return token_ids, token_bytes, cum, F.log_softmax(logits, dim=0), kept_t


def _advance(cur_interval, cum, selection, cfg):
    """Narrow the interval to the selected token and return the settled bits."""
    bottom = int(cum[selection - 1]) if selection > 0 else cur_interval[0]
    top = int(cum[selection])

    bottom_bits = list(reversed(int2bits(bottom, cfg.precision)))
    top_bits = list(reversed(int2bits(top - 1, cfg.precision)))
    n = num_same_from_beg(bottom_bits, top_bits)

    new_bottom = bottom_bits[n:] + [0] * n
    new_top = top_bits[n:] + [1] * n
    cur_interval[0] = bits2int(reversed(new_bottom))
    cur_interval[1] = bits2int(reversed(new_top)) + 1
    return n, top_bits[:n], bottom_bits


@torch.no_grad()
def encode(model, vocab, message_bits, context_ids, cfg, banned_ids, stats=None):
    """Embed `message_bits` into generated text.  Returns (token_ids, stego bytes, stats)."""
    device = cfg.device
    cur_interval = [0, 2 ** cfg.precision]
    stats = stats if stats is not None else StepStats()

    prev = torch.tensor(context_ids, device=device, dtype=torch.long)
    past = None
    out_ids: list[int] = []

    i = 0
    while i < len(message_bits) and len(out_ids) < cfg.max_tokens:
        out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values

        token_ids, _, cum, logp, kept_t = candidate_pool(
            out.logits[0, -1, :], cur_interval, vocab, cfg, banned_ids, stats
        )

        bits = list(message_bits[i:i + cfg.precision])
        bits += [0] * (cfg.precision - len(bits))
        idx = bits2int(reversed(bits))
        hit = (cum > idx).nonzero()
        selection = int(hit[0].item()) if len(hit) else len(cum) - 1

        n, _, _ = _advance(cur_interval, cum, selection, cfg)
        i += n

        stats.logprobs.append(float(logp[kept_t[selection]]))
        stats.steps += 1
        stats.bits_used = min(i, len(message_bits))

        chosen = token_ids[selection]
        out_ids.append(chosen)
        prev = torch.tensor([chosen], device=device, dtype=torch.long)

    return out_ids, vocab.join(out_ids), stats


@torch.no_grad()
def decode(model, vocab, stego_bytes, context_ids, cfg, banned_ids, stats=None):
    """Recover the embedded bits by walking the stego byte stream."""
    device = cfg.device
    cur_interval = [0, 2 ** cfg.precision]
    stats = stats if stats is not None else StepStats()

    prev = torch.tensor(context_ids, device=device, dtype=torch.long)
    past = None
    remaining = bytes(stego_bytes)
    message: list[int] = []
    steps = 0

    while remaining and steps < cfg.max_tokens:
        out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values

        token_ids, token_bytes, cum, _, _ = candidate_pool(
            out.logits[0, -1, :], cur_interval, vocab, cfg, banned_ids, stats
        )

        matches = [j for j, tb in enumerate(token_bytes) if remaining.startswith(tb)]
        if not matches:
            raise ValueError(
                f"desynchronised at step {steps}: no retained candidate is a prefix of "
                f"{remaining[:24]!r}"
            )
        if len(matches) > 1:
            raise AssertionError(
                f"antichain violated at step {steps}: {[token_bytes[j] for j in matches]}"
            )
        selection = matches[0]

        last = len(remaining) == len(token_bytes[selection])
        n, settled, bottom_bits = _advance(cur_interval, cum, selection, cfg)
        message += bottom_bits if last else settled

        remaining = remaining[len(token_bytes[selection]):]
        prev = torch.tensor([token_ids[selection]], device=device, dtype=torch.long)
        steps += 1
        stats.steps += 1

    return message, stats
