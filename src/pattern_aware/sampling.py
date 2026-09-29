from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence, Iterator
import numpy as np
from .config import ContractError

@dataclass(frozen=True)
class PatternBatchAudit:
    eligible_patterns: int
    eligible_members: int
    ineligible_patterns: int
    patterns_per_batch: int
    members_per_pattern: int
    batches: int


class PatternBalancedBatchSampler:
    def __init__(
        self,
        pattern_ids: Sequence[object],
        eligible: Sequence[bool] | np.ndarray,
        *,
        patterns_per_batch: int = 3,
        members_per_pattern: int = 2,
        seed: int = 42,
    ) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2
               for v in (patterns_per_batch, members_per_pattern)):
            raise ContractError("Need at least two patterns and two members")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ContractError("Sampler seed must be a nonnegative integer")

        labels = np.asarray([str(value) for value in pattern_ids], dtype=object)
        available = np.asarray(eligible)
        if available.dtype != np.bool_ or available.shape != labels.shape:
            raise ContractError("Eligibility must be strict Boolean")

        grouped: dict[str, list[int]] = {}
        for index, (label, is_eligible) in enumerate(zip(labels, available, strict=True)):
            if is_eligible:
                grouped.setdefault(label, []).append(index)

        self._members = {
            label: np.asarray(indices, dtype=np.int64)
            for label, indices in sorted(grouped.items())
            if len(indices) >= members_per_pattern
        }
        if len(self._members) < 2:
            raise ContractError("Training needs two eligible multi-member patterns")

        self.patterns_per_batch = patterns_per_batch
        self.members_per_pattern = members_per_pattern
        self.seed = seed
        self.epoch = 0
        self.audit = PatternBatchAudit(
            eligible_patterns=len(self._members),
            eligible_members=sum(len(v) for v in self._members.values()),
            ineligible_patterns=len(set(labels.tolist())) - len(self._members),
            patterns_per_batch=patterns_per_batch,
            members_per_pattern=members_per_pattern,
            batches=len(self),
        )

    def __len__(self) -> int:
        batches = math.ceil(len(self._members) / self.patterns_per_batch)
        if self.patterns_per_batch == 2 and len(self._members) % 2 == 1:
            batches -= 1
        return batches

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ContractError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        labels = np.asarray(list(self._members), dtype=object)
        labels = labels[rng.permutation(len(labels))]
        chunks = [
            labels[start : start + self.patterns_per_batch].tolist()
            for start in range(0, len(labels), self.patterns_per_batch)
        ]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            if len(chunks[-2]) > 2:
                chunks[-1].insert(0, chunks[-2].pop())
            else:
                # With P=2, borrowing creates another singleton. Merge the tail:
                # the final batch has three patterns and still exactly K members each.
                chunks[-2].extend(chunks.pop())

        for pattern_batch in chunks:
            rows: list[int] = []
            for label in pattern_batch:
                selected = rng.choice(
                    self._members[str(label)],
                    size=self.members_per_pattern,
                    replace=False,
                )
                rows.extend(int(index) for index in selected)
            yield rows
