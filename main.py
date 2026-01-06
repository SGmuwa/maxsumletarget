#!/usr/bin/env python3
from __future__ import annotations

from bisect import bisect_right
import argparse
from typing import List, Tuple


def best_subset_dp(nums: List[int], target: int) -> Tuple[int, List[int]]:
    """
    Exact DP: O(n * target) time, O(target) memory.
    Tie-break: among equal sums choose subset that uses larger numbers first.
    Assumes nums >= 0 and target >= 0.
    """
    nums = sorted(nums, reverse=True)
    if target < 0:
        raise ValueError("target must be >= 0")
    if any(x < 0 for x in nums):
        raise ValueError("This DP variant expects all numbers >= 0")

    n = len(nums)
    # Bit positions are chosen so that comparing signatures as integers
    # corresponds to lexicographic compare of chosen items from largest to smallest.
    bits = [1 << (n - 1 - i) for i in range(n)]

    best_sig = [-1] * (target + 1)
    best_sig[0] = 0

    for i, v in enumerate(nums):
        if v > target:
            continue
        bit = bits[i]
        for s in range(target - v, -1, -1):
            sig = best_sig[s]
            if sig == -1:
                continue
            ns = s + v
            nsig = sig | bit
            if best_sig[ns] == -1 or nsig > best_sig[ns]:
                best_sig[ns] = nsig

    best_sum = max(i for i, sig in enumerate(best_sig) if sig != -1)
    sig = best_sig[best_sum]
    subset = [nums[i] for i in range(n) if sig & bits[i]]
    return best_sum, subset


def _enum_half(nums: List[int], bits: List[int], start: int, end: int, target: int) -> dict[int, int]:
    """
    Enumerate all subsets of nums[start:end].
    Return dict: sum -> best_signature (tie-break by signature).
    Only keeps sums <= target.
    """
    pairs = [(0, 0)]  # (sum, signature)
    for i in range(start, end):
        v, b = nums[i], bits[i]
        pairs += [(s + v, sig | b) for s, sig in pairs]

    best = {}
    for s, sig in pairs:
        if s <= target:
            prev = best.get(s)
            if prev is None or sig > prev:
                best[s] = sig
    return best


def best_subset_mitm(nums: List[int], target: int) -> Tuple[int, List[int]]:
    """
    Exact meet-in-the-middle: good when n is small (≈ up to 40-44) and target is huge.
    Time ~ O(2^(n/2)), memory ~ O(2^(n/2)).
    """
    nums = sorted(nums, reverse=True)
    if target < 0:
        raise ValueError("target must be >= 0")
    if any(x < 0 for x in nums):
        raise ValueError("This MITM variant expects all numbers >= 0")

    n = len(nums)
    bits = [1 << (n - 1 - i) for i in range(n)]
    mid = n // 2

    left = _enum_half(nums, bits, 0, mid, target)
    right = _enum_half(nums, bits, mid, n, target)

    right_items = sorted(right.items())  # list of (sum, best_sig_for_that_sum)
    sums_r = [s for s, _ in right_items]

    best_total = -1
    best_sig = -1

    for sum_l, sig_l in left.items():
        cap = target - sum_l
        idx = bisect_right(sums_r, cap) - 1
        if idx < 0:
            continue
        sum_r, sig_r = right_items[idx]
        total = sum_l + sum_r
        sig = sig_l | sig_r
        if total > best_total or (total == best_total and sig > best_sig):
            best_total = total
            best_sig = sig

    subset = [nums[i] for i in range(n) if best_sig & bits[i]]
    return best_total, subset


def solve(nums: List[int], target: int, algo: str = "auto") -> Tuple[int, List[int]]:
    """
    algo:
      - dp: exact, best when target is moderate
      - mitm: exact, best when n is small and target is huge
      - auto: pick dp if feasible, else mitm if feasible, else raise
    """
    n = len(nums)
    if algo == "dp":
        return best_subset_dp(nums, target)
    if algo == "mitm":
        return best_subset_mitm(nums, target)

    # auto:
    # Heuristics to avoid accidental huge allocations / runtimes.
    # Tune thresholds to your data.
    dp_ops = n * max(0, target)
    if target <= 2_000_000 and dp_ops <= 50_000_000:
        return best_subset_dp(nums, target)
    if n <= 44:
        return best_subset_mitm(nums, target)

    raise RuntimeError(
        "Слишком большие входные данные для точного решения без внешнего солвера. "
        "Либо уменьшите target/кол-во чисел, либо используйте OR-Tools CP-SAT."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find subset with max sum <= target; tie-break prefers larger numbers first."
    )
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--algo", choices=["auto", "dp", "mitm"], default="auto")
    p.add_argument("nums", nargs="+", type=int, help="List of numbers (each can be used at most once)")
    args = p.parse_args()

    best_sum, subset = solve(args.nums, args.target, args.algo)
    expr = "+".join(map(str, subset)) if subset else "0"
    print(
        '\n'.join(
            [
                f"target = {args.target}",
                f"best_sum = {best_sum}",
                f"target - best_sum = {args.target - best_sum}",
                f"subset = {expr}",
            ]
        )
    )


if __name__ == "__main__":
    main()
