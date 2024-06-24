from __future__ import annotations

from typing import Dict, Tuple


class CountStore:
    def __init__(self, prevctx_width: int):
        self.counts_ordered: Dict[Tuple, Dict[int, int]] = {}
        self.counts_unordered: Dict[Tuple, Dict[int, int]] = {}
        self.prevctx_width = prevctx_width

        # Generate all subsets (used for both ordered and unordered)
        self.masks = []  # 1 for select, -1 for wildcard/ignore
        for mask in range(2**prevctx_width):
            self.masks.append([1 if mask & (1 << i) else -1 for i in range(self.prevctx_width)])

    def _add_to_dict(
        self, dic: Dict[Tuple, Dict[int, int]], ctx: Tuple, tok: int, quantity: int
    ) -> None:
        if ctx not in dic:
            dic[ctx] = {}
        if tok not in dic[ctx]:
            dic[ctx][tok] = 0
        dic[ctx][tok] += quantity

    # Context should be (tokId|-1){3}, i.e. it's a []
    def _get_ordered(self, ctx: Tuple) -> Dict[int, int]:
        if len(ctx) != self.prevctx_width:
            raise ValueError(
                f"Context length of {ctx} does not match prevctx_width of {self.prevctx_width}"
            )
        return self.counts_ordered.get(ctx, {})

    # Context [ctx] should be (tokId){0,3}, sorted, i.e. it's a {}
    def _get_unordered(self, ctx: Tuple) -> Dict[int, int]:
        if len(ctx) > self.prevctx_width:
            raise ValueError(
                f"Context length of {ctx} does not match prevctx_width of {self.prevctx_width}"
            )
        if ctx != tuple(sorted(list(ctx))):
            raise ValueError(f"{ctx} is not sorted for unordered mode")
        if -1 in ctx:
            raise ValueError(f"-1 in ctx {ctx} invalid in unordered mode")
        return self.counts_unordered.get(ctx, {})

    # You can only add proper (A,B,C,D) tuples; the rest is filled internally
    def add(self, ctx: Tuple, tok: int, quantity: int) -> None:
        # Ctx can be empty for GPTWM
        if len(ctx) != self.prevctx_width:
            raise ValueError(
                f"Context length of {ctx} does not match prevctx_width of {self.prevctx_width}"
            )
        if len(ctx) > 0 and min(ctx) < 0:
            raise ValueError(f"Context {ctx} in .add contains negative values")

        # Add all
        for mask in self.masks:
            ctx_ord = tuple([ctx[i] if mask[i] == 1 else -1 for i in range(self.prevctx_width)])
            self._add_to_dict(self.counts_ordered, ctx_ord, tok, quantity)
            ctx_uord = tuple(sorted([ctx[i] for i in range(self.prevctx_width) if mask[i] == 1]))
            self._add_to_dict(self.counts_unordered, ctx_uord, tok, quantity)

    def get(self, ctx: Tuple, ordered: bool) -> Dict[int, int]:
        if ordered:
            return self._get_ordered(ctx)
        else:
            return self._get_unordered(ctx)

    def nb_keys(self, ordered: bool) -> int:
        counts = self.counts_ordered if ordered else self.counts_unordered
        return len(counts)

    def total_nb_counts(self, ordered: bool) -> int:
        counts = self.counts_ordered if ordered else self.counts_unordered
        return sum([sum(dic.values()) for dic in counts.values()])

    def update(self, other: CountStore) -> None:
        self.counts_ordered.update(other.counts_ordered)
        self.counts_unordered.update(other.counts_unordered)
        # buggy for LeftHash
        # assert self.prevctx_width == other.prevctx_width

    def clear(self) -> None:
        self.counts_ordered = {}
        self.counts_unordered = {}
