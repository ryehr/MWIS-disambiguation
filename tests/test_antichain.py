"""The exact solver must match brute-force MWIS on random prefix-structured pools."""
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwis_stego.antichain import (
    exact, exact_trie, greedy, enumerate_cc, brute_force, is_antichain,
)


def random_pool(rng, n, alphabet=b"abc", maxlen=4):
    """Pools with heavy prefix structure: short alphabet, short tokens."""
    toks = set()
    while len(toks) < n:
        L = rng.randint(1, maxlen)
        toks.add(bytes(rng.choice(alphabet) for _ in range(L)))
    toks = sorted(toks)
    rng.shuffle(toks)
    weights = [rng.randint(1, 1000) for _ in toks]
    return toks, weights


def weight(w, kept):
    return sum(w[i] for i in kept)


def main():
    rng = random.Random(0)
    trials = 3000
    greedy_losses = 0
    greedy_gap = 0.0
    for t in range(trials):
        n = rng.randint(1, 12)
        toks, w = random_pool(rng, n)

        opt = weight(w, brute_force(toks, w))
        e = exact(toks, w)
        et = exact_trie(toks, w)
        c = enumerate_cc(toks, w)
        g = greedy(toks, w)

        assert is_antichain(toks, e), f"exact produced a conflict: {toks} {e}"
        assert is_antichain(toks, c), f"enumerate produced a conflict: {toks} {c}"
        assert is_antichain(toks, g), f"greedy produced a conflict: {toks} {g}"

        # The two exact solvers must agree token for token, not merely in weight:
        # both sides of the channel must retain the identical pool.
        assert e == et, f"exact and exact_trie disagree: {toks} {w}\n  {e}\n  {et}"

        assert weight(w, e) == opt, f"exact suboptimal: {toks} {w} got {weight(w,e)} want {opt}"
        assert weight(w, c) == opt, f"enumerate suboptimal: {toks} {w}"
        assert weight(w, g) <= opt

        if weight(w, g) < opt:
            greedy_losses += 1
            greedy_gap += 1 - weight(w, g) / opt

    print(f"{trials} random pools: exact == enumerate == brute-force optimum")
    print(f"greedy strictly suboptimal on {greedy_losses}/{trials} pools "
          f"({100*greedy_losses/trials:.1f}%), mean eta shortfall on those "
          f"{100*greedy_gap/max(greedy_losses,1):.1f}%")

    # Degenerate inputs
    assert exact([], []) == []
    assert exact([b"a"], [5]) == [0]
    assert exact([b"a", b"a"], [5, 7]) == [1], "identical tokens conflict; keep the heavier"
    assert exact([b"a", b"b"], [1, 1]) == [0, 1], "no prefix relation -> keep everything"
    assert exact([b"a", b"ab", b"abc"], [3, 2, 2]) == [0], "a chain admits a single element"
    assert exact([b"a", b"ab", b"ac"], [3, 2, 2]) == [1, 2], "branch: the two leaves outweigh the root"
    assert exact([b"a", b"ab", b"ac"], [9, 2, 2]) == [0], "branch: the root outweighs its leaves"
    print("degenerate cases OK")


if __name__ == "__main__":
    main()
