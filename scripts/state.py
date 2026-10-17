"""Per-round bookkeeping in ``loop_state.json``.

The state file is the loop's memory. It records, for each round, which checkpoint pre-labelled
which images, the metrics at that time, and the convergence measured after the human corrected
the pre-labels. It is small JSON and lives in the code repo so the loop can resume across runs.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoundRecord:
    """Everything recorded for one round.

    Attributes
    ----------
    round_number : int
        Round identifier.
    train_project : str
        Project the round trained on.
    prelabel_project : str
        Project the round pre-labelled.
    prelabel_class : str
        Class written on the pre-labels.
    checkpoint : str
        Path to the checkpoint, relative to the repository when possible.
    metrics : dict of str to object
        Training metrics for the checkpoint.
    uploaded : int
        Number of images that received a pre-label.
    skipped : int
        Number of target images skipped because they were already annotated.
    uploaded_images : list of str
        Image ids that received a pre-label, so a later round can measure convergence.
    convergence : dict of str to object or None
        Convergence measured after correction, or ``None`` before correction.
    timestamp : str
        When the round was recorded.
    """

    round_number: int
    train_project: str
    prelabel_project: str
    prelabel_class: str
    checkpoint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    uploaded: int = 0
    skipped: int = 0
    uploaded_images: list[str] = field(default_factory=list)
    convergence: dict[str, Any] | None = None
    timestamp: str = ""


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class LoopState:
    """The whole loop state: ordered round records.

    Attributes
    ----------
    rounds : list of RoundRecord
        The rounds, oldest first.
    """

    rounds: list[RoundRecord] = field(default_factory=list)

    def upsert(self, record: RoundRecord) -> RoundRecord:
        """Insert or replace the record for a round.

        Parameters
        ----------
        record : RoundRecord
            The record to store. An existing record for the same round is replaced, keeping
            any convergence already measured unless the new record carries its own.

        Returns
        -------
        RoundRecord
            The stored record.
        """
        if not record.timestamp:
            record.timestamp = _now()
        for index, existing in enumerate(self.rounds):
            if existing.round_number == record.round_number:
                if record.convergence is None and existing.convergence is not None:
                    record.convergence = existing.convergence
                self.rounds[index] = record
                return record
        self.rounds.append(record)
        self.rounds.sort(key=lambda r: r.round_number)
        return record

    def for_round(self, round_number: int) -> RoundRecord | None:
        """Return the record for a round, or ``None``.

        Parameters
        ----------
        round_number : int
            The round to look up.

        Returns
        -------
        RoundRecord or None
            The record, if present.
        """
        for record in self.rounds:
            if record.round_number == round_number:
                return record
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the state.

        Returns
        -------
        dict of str to object
            A mapping with a single ``rounds`` list.
        """
        return {"rounds": [asdict(record) for record in self.rounds]}


def load_state(path: Path) -> LoopState:
    """Load the loop state, returning an empty state when the file is absent.

    Parameters
    ----------
    path : Path
        Path to ``loop_state.json``.

    Returns
    -------
    LoopState
        The loaded state.

    Raises
    ------
    ValueError
        When the file exists but is not a JSON object with a ``rounds`` list.
    """
    if not path.is_file():
        return LoopState()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rounds"), list):
        raise ValueError(f"state file is malformed: {path}")
    rounds: list[RoundRecord] = []
    for entry in data["rounds"]:
        if not isinstance(entry, dict):
            raise ValueError(f"state file has a non-object round: {path}")
        rounds.append(
            RoundRecord(
                round_number=int(entry.get("round_number", 0)),
                train_project=str(entry.get("train_project", "")),
                prelabel_project=str(entry.get("prelabel_project", "")),
                prelabel_class=str(entry.get("prelabel_class", "")),
                checkpoint=str(entry.get("checkpoint", "")),
                metrics=dict(entry.get("metrics") or {}),
                uploaded=int(entry.get("uploaded", 0)),
                skipped=int(entry.get("skipped", 0)),
                uploaded_images=[str(x) for x in (entry.get("uploaded_images") or [])],
                convergence=entry.get("convergence"),
                timestamp=str(entry.get("timestamp", "")),
            )
        )
    return LoopState(rounds=rounds)


def save_state(path: Path, state: LoopState) -> None:
    """Write the loop state as indented JSON.

    Parameters
    ----------
    path : Path
        Destination path; its parent directory is created.
    state : LoopState
        The state to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
