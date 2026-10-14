"""Deterministic, resumable multi-epoch traversal over a fixed set of unique example IDs.

Standard library only. Nothing here knows about models, tensors, files, labels or
the dataset size: the runner passes the frozen training IDs, a seed, the number of
examples per optimizer step and the number of epochs.

Order. Within epoch ``e`` (0-based) the IDs are sorted by
``sha256(canonical JSON [SCHEME, seed, e, id])``, with the ID itself as the
tie-breaker. The order depends only on (seed, e, set of IDs). It does not depend
on the input order, Python's ``random`` module, hash randomization or the number
of epochs. Note that the JSON encoding distinguishes the int seed ``42`` from the
str seed ``"42"``.

Steps. Each epoch is cut into consecutive optimizer steps of ``batch_size`` IDs,
and steps never cross an epoch boundary. If the epoch size is not a multiple of
``batch_size``, the last step of every epoch is a smaller final partial step
holding the remainder. Nothing is dropped, padded or borrowed from the next
epoch, so every ID is used exactly once per epoch.

Position. A run's position is the number of completed optimizer steps
(``global_step``). ``locate`` maps it to the epoch and step-in-epoch of the next
step, and ``steps(..., start_step=global_step)`` resumes the identical sequence.
``audit`` checks an executed sequence against the expected one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

SCHEME = "goqueen-epoch-sha256-v1"


def validate_ids(example_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the IDs in canonical sorted order; reject empty, non-string or duplicate IDs."""
    ids = list(example_ids)
    if not ids:
        raise ValueError("at least one example ID is required")
    for index, value in enumerate(ids):
        if not isinstance(value, str) or not value:
            raise ValueError(f"example ID {index} must be a non-empty string")
    counts: dict[str, int] = {}
    for value in ids:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) != len(ids):
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        raise ValueError(f"{len(ids) - len(counts)} duplicate example IDs, e.g. {duplicates[:3]!r}")
    return tuple(sorted(ids))


def _check_seed(seed) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise TypeError("seed must be an int or a str")


def _check_count(name: str, value, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Plan:
    """Counts for one fixed set of examples traversed a fixed number of times."""
    examples: int
    batch_size: int
    epochs: int
    steps_per_epoch: int
    final_step_size: int  # size of the last step of every epoch; equals batch_size if divisible
    total_steps: int
    example_passes: int   # examples * epochs: every example once per epoch


def make_plan(examples: int, batch_size: int, epochs: int) -> Plan:
    _check_count("examples", examples, 1)
    _check_count("batch_size", batch_size, 1)
    _check_count("epochs", epochs, 1)
    steps_per_epoch = -(-examples // batch_size)
    final_step_size = examples - (steps_per_epoch - 1) * batch_size
    return Plan(examples, batch_size, epochs, steps_per_epoch, final_step_size,
                steps_per_epoch * epochs, examples * epochs)


def _sort_key(seed, epoch: int, example_id: str) -> tuple[str, str]:
    payload = json.dumps([SCHEME, seed, epoch, example_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), example_id


def epoch_order(example_ids: Iterable[str], seed, epoch: int) -> list[str]:
    """The full order of epoch ``epoch`` (0-based)."""
    ids = validate_ids(example_ids)
    _check_seed(seed)
    _check_count("epoch", epoch, 0)
    return sorted(ids, key=lambda value: _sort_key(seed, epoch, value))


def epoch_steps(example_ids: Iterable[str], seed, epoch: int, batch_size: int) -> list[tuple[str, ...]]:
    """Epoch ``epoch`` cut into optimizer steps; only the last step may be smaller."""
    _check_count("batch_size", batch_size, 1)
    order = epoch_order(example_ids, seed, epoch)
    return [tuple(order[start:start + batch_size]) for start in range(0, len(order), batch_size)]


@dataclass(frozen=True)
class Position:
    global_step: int     # completed optimizer steps
    epoch: int           # 0-based epoch of the next step; equals plan.epochs when finished
    step_in_epoch: int   # 0-based index of the next step within its epoch; 0 when finished
    finished: bool


def locate(plan: Plan, global_step: int) -> Position:
    _check_count("global_step", global_step, 0)
    if global_step > plan.total_steps:
        raise ValueError(f"global_step {global_step} exceeds the plan's {plan.total_steps} steps")
    if global_step == plan.total_steps:
        return Position(global_step, plan.epochs, 0, True)
    epoch, step_in_epoch = divmod(global_step, plan.steps_per_epoch)
    return Position(global_step, epoch, step_in_epoch, False)


@dataclass(frozen=True)
class Step:
    global_step: int          # 0-based index of this optimizer step
    epoch: int                # 0-based
    step_in_epoch: int        # 0-based
    example_ids: tuple[str, ...]
    final_in_epoch: bool


def steps(example_ids: Iterable[str], seed, batch_size: int, epochs: int,
          start_step: int = 0) -> Iterator[Step]:
    """All remaining optimizer steps from ``start_step`` (completed steps) to the end.

    Arguments are validated before the iterator is returned. Each epoch's order is
    recomputed from (seed, epoch, IDs), so a resumed iterator yields exactly the
    remainder of an uninterrupted one.
    """
    ids = validate_ids(example_ids)
    _check_seed(seed)
    plan = make_plan(len(ids), batch_size, epochs)
    start = locate(plan, start_step)
    return _generate(ids, seed, plan, start)


def _generate(ids: tuple[str, ...], seed, plan: Plan, start: Position) -> Iterator[Step]:
    for epoch in range(start.epoch, plan.epochs):
        batches = epoch_steps(ids, seed, epoch, plan.batch_size)
        first = start.step_in_epoch if epoch == start.epoch else 0
        for index in range(first, len(batches)):
            yield Step(epoch * plan.steps_per_epoch + index, epoch, index, batches[index],
                       index == len(batches) - 1)


def fingerprint(example_ids: Iterable[str], seed, batch_size: int, epochs: int) -> str:
    """Identity of the whole schedule, for checkpoints and resume checks."""
    ids = validate_ids(example_ids)
    _check_seed(seed)
    make_plan(len(ids), batch_size, epochs)
    ids_digest = order_digest(ids)
    payload = json.dumps([SCHEME, seed, batch_size, epochs, len(ids), ids_digest],
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def order_digest(order: Iterable[str]) -> str:
    """Digest of one realized sequence of IDs, e.g. one epoch's order, for reports."""
    return hashlib.sha256(
        json.dumps(list(order), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit(executed: Sequence[Sequence[str]], example_ids: Iterable[str], seed,
          batch_size: int, epochs: int) -> dict:
    """Compare executed optimizer steps (ID tuples, in execution order) with the schedule.

    Raises ValueError at the first step that differs, or if more steps were executed
    than planned. Otherwise reports how far the run got; ``complete`` is true only
    when every epoch was traversed in full.
    """
    ids = validate_ids(example_ids)
    _check_seed(seed)
    plan = make_plan(len(ids), batch_size, epochs)
    executed = [tuple(step) for step in executed]
    if len(executed) > plan.total_steps:
        raise ValueError(f"{len(executed)} executed steps exceed the plan's {plan.total_steps}")
    for expected, actual in zip(steps(ids, seed, batch_size, epochs), executed):
        if actual != expected.example_ids:
            raise ValueError(f"step {expected.global_step} (epoch {expected.epoch}, "
                             f"step {expected.step_in_epoch}) differs from the schedule")
    position = locate(plan, len(executed))
    return {"scheme": SCHEME, "fingerprint": fingerprint(ids, seed, batch_size, epochs),
            "steps_completed": len(executed), "total_steps": plan.total_steps,
            "epochs_completed": len(executed) // plan.steps_per_epoch,
            "next_epoch": position.epoch, "next_step_in_epoch": position.step_in_epoch,
            "complete": position.finished}
