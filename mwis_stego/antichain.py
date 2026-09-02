"""Token-selection disambiguation on a candidate pool.

The candidate pool of a language model is a set of tokens, each a *byte* string.
Two candidates are in conflict when one is a prefix of the other, because the
receiver, scanning the remaining stegatext, would then match both.  Removing
conflicts means picking an independent set in the graph

    E = {(c_i, c_j) : c_i is a prefix of c_j, i != j}

and picking a *maximum weight* independent set (MWIS) minimises the KL
divergence between the retained pool and the model's own distribution: with
eta = sum of retained probabilities, D_KL(CP_a || CP) = -log(eta).

Structure of the problem
------------------------
"Is a prefix of" is a partial order, and the Hasse diagram of that order is a
*forest*: a candidate's parent is its longest proper prefix that is also a
candidate, and that is unique.  Two candidates conflict exactly when they are
ancestor/descendant in this forest, so the conflict graph is the comparability
graph of a forest poset, and an independent set in it is an *antichain*.

Maximum weight antichains in a forest are computable exactly in linear time,
so this MWIS instance is not NP-hard.  `exact` below solves it in
O(sum of candidate byte lengths) via a trie; `greedy` and `enumerate_cc` are
the two approximations from the original paper, kept as baselines.

Weights are integers (quantised probabilities).  Integer weights make every
comparison exact, so the sender and the receiver always derive the identical
retained pool -- with floats, a one-ULP difference could silently desynchronise
extraction.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

__all__ = ["exact", "exact_trie", "greedy", "enumerate_cc", "brute_force",
           "conflict_edges", "is_antichain", "EnumerationInfeasible"]


class EnumerationInfeasible(RuntimeError):
    """The paper's 2^|C| baseline hit a component too large to enumerate.

    Not a defect: it is the measured failure mode of that baseline on real
    byte-BPE pools, where a whole top-k pool can form one connected component.
    """

    def __init__(self, size: int):
        self.size = size
        super().__init__(f"connected component of size {size} exceeds the enumeration budget")


# --------------------------------------------------------------------------
# Exact: maximum weight antichain of the prefix forest, O(sum |c_i|)
# --------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "terms", "own", "best", "take_own")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.terms: list[int] = []   # candidate indices ending here
        self.own = 0                 # best single weight among `terms`
        self.best = 0                # best antichain weight in this subtree
        self.take_own = False        # reconstruction decision


def exact(tokens: Sequence[bytes], weights: Sequence[int]) -> list[int]:
    """Maximum weight antichain of the prefix order, exactly.

    Same result as `exact_trie`, without materialising the trie.

    Sorting the candidates lexicographically makes the prefix order explicit:
    if `t_i` is a prefix of `t_j` then `i < j`, and every candidate between them
    also starts with `t_i`.  Two consequences:

    * A pool contains a prefix conflict **iff some adjacent pair does**, so one
      O(k) scan decides it.  Real pools are conflict-free 10-35% of the time and
      those return immediately.
    * Parent pointers of the prefix forest fall out of a single stack pass, with
      no per-node object or dict.  That is the whole cost of the trie version in
      Python, and pools here are small enough (k <= 128, a few bytes each) that
      the constant factor decides, not the asymptotics.

    Ties are broken towards the *descendants* (a node's own weight wins only when
    strictly heavier), which retains more tokens at equal eta and so costs no
    embedding rate.  The rule must be deterministic, not merely reasonable: the
    sender and receiver run this independently and must agree exactly.
    """
    k = len(tokens)
    if k <= 1:
        return list(range(k))

    order = sorted(range(k), key=tokens.__getitem__)
    sorted_toks = [tokens[i] for i in order]

    # Fast path: no adjacent pair conflicts => no pair conflicts at all.
    conflict = False
    for j in range(k - 1):
        if sorted_toks[j + 1].startswith(sorted_toks[j]):
            conflict = True
            break
    if not conflict:
        return list(range(k))

    # Parent pointers of the prefix forest, in one stack pass.
    parent = [-1] * k
    stack: list[int] = []
    for j in range(k):
        tok = sorted_toks[j]
        while stack and not tok.startswith(sorted_toks[stack[-1]]):
            stack.pop()
        parent[j] = stack[-1] if stack else -1
        stack.append(j)

    # Bottom-up DP.  Children always have a larger index than their parent, so a
    # single reverse pass sees every child before its parent.
    w = [weights[i] for i in order]
    below = [0] * k
    take = [False] * k
    for j in range(k - 1, -1, -1):
        own = w[j]
        if own > below[j]:
            best, take[j] = own, True
        else:
            best = below[j]
        p = parent[j]
        if p >= 0:
            below[p] += best

    # Descend: keep a node if it is taken and no ancestor was taken before it.
    kept: list[int] = []
    reachable = [False] * k
    for j in range(k):
        p = parent[j]
        reachable[j] = True if p < 0 else (reachable[p] and not take[p])
        if reachable[j] and take[j]:
            kept.append(order[j])
    kept.sort()
    return kept


def exact_trie(tokens: Sequence[bytes], weights: Sequence[int]) -> list[int]:
    """Reference implementation of `exact`, building the trie explicitly.

    Kept because it mirrors the recurrence in the write-up one-to-one and so is
    the clearer statement of *why* the answer is optimal; `exact` is the same
    computation with the trie left implicit.  The tests assert they agree.
    """
    if not tokens:
        return []

    root = _TrieNode()
    for i, tok in enumerate(tokens):
        node = root
        for b in tok:
            nxt = node.children.get(b)
            if nxt is None:
                nxt = _TrieNode()
                node.children[b] = nxt
            node = nxt
        node.terms.append(i)

    # Identical byte strings are prefixes of each other, so at most one of a
    # duplicate group may survive: keep the heaviest, lowest index on a tie.
    order: list[_TrieNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(node.children.values())

    for node in reversed(order):  # children precede parents
        node.own = max((weights[i] for i in node.terms), default=0)
        below = 0
        for child in node.children.values():
            below += child.best
        if node.terms and node.own > below:
            node.best = node.own
            node.take_own = True
        else:
            node.best = below
            node.take_own = False

    kept: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.take_own:
            best_i = min(node.terms, key=lambda i: (-weights[i], i))
            kept.append(best_i)
        else:
            stack.extend(node.children.values())
    kept.sort()
    return kept


# --------------------------------------------------------------------------
# Baselines from the original paper
# --------------------------------------------------------------------------

def conflict_edges(tokens: Sequence[bytes]) -> list[list[int]]:
    """Adjacency by pairwise `startswith`, as in the published implementation.

    Kept O(n^2) on purpose: it is what the paper's running-time table measures.
    """
    n = len(tokens)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (tokens[i].startswith(tokens[j]) or tokens[j].startswith(tokens[i])):
                adj[i].append(j)
    return adj


def greedy(tokens: Sequence[bytes], weights: Sequence[int]) -> list[int]:
    """Paper Algorithm 2: extract a spanning forest, then DP over BFS layers.

    A node joins the forest when at most one of its lower-indexed neighbours is
    already in it, which makes the induced subgraph acyclic.  Within one tree,
    nodes in non-adjacent BFS layers are never adjacent, so a weighted
    "no two consecutive" DP over layers yields an independent set.  Nodes left
    out of the forest are never reconsidered, hence the approximation.
    """
    n = len(tokens)
    if n == 0:
        return []
    adj = conflict_edges(tokens)

    forest: list[int] = []
    in_forest = [False] * n
    for i in range(n):
        seen = sum(1 for nb in adj[i] if nb < i)
        if seen <= 1:
            forest.append(i)
            in_forest[i] = True

    kept: list[int] = []
    remaining = set(forest)
    while remaining:
        start = min(remaining)
        layers: list[list[int]] = []
        visited = {start}
        frontier = [start]
        while frontier:
            layers.append(frontier)
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if in_forest[v] and v not in visited:
                        visited.add(v)
                        nxt.append(v)
            frontier = nxt
        remaining -= visited

        layer_w = [sum(weights[i] for i in layer) for layer in layers]
        m = len(layer_w)
        dp = [0] * (m + 1)
        for k in range(1, m + 1):
            dp[k] = max(dp[k - 1], layer_w[k - 1] + (dp[k - 2] if k >= 2 else 0))
        k = m
        while k >= 1:
            if dp[k] == dp[k - 1]:
                k -= 1
            else:
                kept.extend(layers[k - 1])
                k -= 2
    kept.sort()
    return kept


def _components(adj: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(adj)
    seen = [False] * n
    comps: list[list[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        comp = [s]
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    queue.append(v)
        comps.append(sorted(comp))
    return comps


def enumerate_cc(tokens: Sequence[bytes], weights: Sequence[int], max_component: int = 22) -> list[int]:
    """Paper's exhaustive baseline: 2^|C| enumeration per connected component."""
    n = len(tokens)
    if n == 0:
        return []
    adj = conflict_edges(tokens)
    adj_set = [set(a) for a in adj]

    kept: list[int] = []
    for comp in _components(adj):
        size = len(comp)
        if size > max_component:
            raise EnumerationInfeasible(size)
        best_w, best_mask = -1, 0
        for mask in range(1 << size):
            ok = True
            total = 0
            for bit in range(size):
                if not (mask >> bit) & 1:
                    continue
                u = comp[bit]
                for other in range(size):
                    if other != bit and (mask >> other) & 1 and comp[other] in adj_set[u]:
                        ok = False
                        break
                if not ok:
                    break
                total += weights[u]
            if ok and total > best_w:
                best_w, best_mask = total, mask
        kept.extend(comp[bit] for bit in range(size) if (best_mask >> bit) & 1)
    kept.sort()
    return kept


def brute_force(tokens: Sequence[bytes], weights: Sequence[int]) -> list[int]:
    """Reference MWIS over all 2^n subsets.  Tests only."""
    n = len(tokens)
    if n > 20:
        raise ValueError("brute_force is for tiny instances only")
    adj_set = [set(a) for a in conflict_edges(tokens)]
    best_w, best = -1, []
    for mask in range(1 << n):
        chosen = [i for i in range(n) if (mask >> i) & 1]
        if any(v in adj_set[u] for u in chosen for v in chosen if u != v):
            continue
        total = sum(weights[i] for i in chosen)
        if total > best_w:
            best_w, best = total, chosen
    return best


def is_antichain(tokens: Sequence[bytes], kept: Sequence[int]) -> bool:
    """No retained token is a prefix of another -- the disambiguation invariant."""
    picked = [tokens[i] for i in kept]
    return not any(
        a.startswith(b) or b.startswith(a)
        for x, a in enumerate(picked)
        for y, b in enumerate(picked)
        if x != y
    )


SOLVERS = {"exact": exact, "greedy": greedy, "enumerate": enumerate_cc}
