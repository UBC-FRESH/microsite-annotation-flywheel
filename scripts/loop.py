#!/usr/bin/env python3
"""The operator command line for the annotation flywheel.

Four boring commands, one per stage of the loop, plus ``compare`` to measure convergence:

    python scripts/loop.py sync       # download the latest annotations
    python scripts/loop.py train      # train the pre-labeller
    python scripts/loop.py prelabel   # upload editable pre-labels
    python scripts/loop.py compare    # measure how much the human edited
    python scripts/loop.py status     # show the per-round trend

The commands are thin wrappers over the importable functions in this package. Configuration
comes from ``config.yaml``; the Roboflow key comes from the ``ROBOFLOW_API_KEY`` environment
variable or a gitignored ``.env`` next to this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import Config, load_config
from convergence import aggregate, image_convergence, trend_table
from geometry import DetectedObject
from prelabel import run_prelabel
from roboflow_api import RoboflowClient
from state import LoopState, RoundRecord, load_state, save_state
from sync import run_sync
from train import run_train

ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = ROOT / "config.yaml"
DEFAULT_ENV: Path = ROOT / ".env"


def read_dotenv(path: Path) -> dict[str, str]:
    """Read simple ``KEY=VALUE`` lines from a dotenv file.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    dict of str to str
        The parsed pairs; empty when the file is absent. Blank lines and ``#`` comments are
        ignored, and surrounding quotes are stripped.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(repo_root: Path, environ: dict[str, str] | None = None) -> str:
    """Return the Roboflow API key from the environment or ``.env``.

    Parameters
    ----------
    repo_root : Path
        Repository root that may hold a ``.env``.
    environ : dict of str to str or None, optional
        Environment mapping; defaults to ``os.environ``.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    SystemExit
        When no key is found.
    """
    import os

    source = environ if environ is not None else dict(os.environ)
    key = source.get("ROBOFLOW_API_KEY")
    if not key:
        key = read_dotenv(repo_root / ".env").get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Put it in a gitignored .env as "
            "ROBOFLOW_API_KEY=... or export it in your shell."
        )
    return key


def build_client(config: Config, repo_root: Path) -> RoboflowClient:
    """Build a Roboflow client for a configuration.

    Parameters
    ----------
    config : Config
        The flywheel configuration.
    repo_root : Path
        Repository root, used to find ``.env``.

    Returns
    -------
    RoboflowClient
        The client.
    """
    return RoboflowClient(resolve_api_key(repo_root), config.workspace)


def _update_state(
    config: Config,
    *,
    checkpoint: str = "",
    metrics: dict[str, object] | None = None,
    uploaded: int = 0,
    skipped: int = 0,
    uploaded_images: list[str] | None = None,
) -> RoundRecord:
    """Load, update, and save the round record.

    Parameters
    ----------
    config : Config
        The flywheel configuration.
    checkpoint : str, optional
        Checkpoint path to record.
    metrics : dict of str to object or None, optional
        Metrics to record.
    uploaded : int, optional
        Uploaded image count.
    skipped : int, optional
        Skipped image count.
    uploaded_images : list of str or None, optional
        Uploaded image ids.

    Returns
    -------
    RoundRecord
        The stored record.
    """
    state = load_state(config.state_path)
    record = state.for_round(config.round_number) or RoundRecord(
        round_number=config.round_number,
        train_project=config.train_project,
        prelabel_project=config.prelabel_project,
        prelabel_class=config.prelabel_class,
    )
    if checkpoint:
        record.checkpoint = checkpoint
    if metrics is not None:
        record.metrics = metrics
    if uploaded:
        record.uploaded = uploaded
    if skipped:
        record.skipped = skipped
    if uploaded_images is not None:
        record.uploaded_images = uploaded_images
    state.upsert(record)
    save_state(config.state_path, state)
    return record


def cmd_sync(config: Config, args: argparse.Namespace) -> int:
    """Run the ``sync`` command.

    Parameters
    ----------
    config : Config
        The configuration.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """
    client = build_client(config, ROOT)
    result = run_sync(config, client, limit=args.limit)
    _update_state(config, uploaded_images=None)
    print(json.dumps({"annotated": result.annotated, "per_split": result.per_split}, indent=2))
    return 0


def cmd_train(config: Config, args: argparse.Namespace) -> int:
    """Run the ``train`` command.

    Parameters
    ----------
    config : Config
        The configuration.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """
    result = run_train(config, encoder_weights=None if args.random_init else "imagenet")
    _update_state(
        config,
        checkpoint=str(result.checkpoint_path),
        metrics={
            "best_val_iou": result.best_val_iou,
            "test_iou": result.test_iou,
            "best_epoch": result.best_epoch,
        },
    )
    print(
        json.dumps(
            {"checkpoint": str(result.checkpoint_path), "test_iou": result.test_iou}, indent=2
        )
    )
    return 0


def cmd_prelabel(config: Config, args: argparse.Namespace) -> int:
    """Run the ``prelabel`` command.

    Parameters
    ----------
    config : Config
        The configuration.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.checkpoints_dir / "best.pt"
    result = run_prelabel(config, build_client(config, ROOT), checkpoint, limit=args.limit)
    _update_state(
        config,
        checkpoint=str(checkpoint),
        uploaded=result.uploaded,
        skipped=result.skipped,
        uploaded_images=result.uploaded_images,
    )
    print(json.dumps({"uploaded": result.uploaded, "skipped": result.skipped}, indent=2))
    return 0


def cmd_compare(config: Config, args: argparse.Namespace) -> int:
    """Run the ``compare`` command: measure convergence for a round.

    Parameters
    ----------
    config : Config
        The configuration.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """
    uploads_path = config.predictions_dir / f"uploads-round-{config.round_number}.json"
    if not uploads_path.is_file():
        print(f"no upload record for round {config.round_number}: {uploads_path}")
        return 1
    uploads = json.loads(uploads_path.read_text(encoding="utf-8"))
    client = build_client(config, ROOT)
    per_image: list[dict[str, float | int]] = []
    for record in uploads.get("uploaded", []):
        detail = client.image_detail(config.prelabel_project, record["image_id"])
        uploaded = [
            DetectedObject(
                label=config.prelabel_class, points=tuple((float(x), float(y)) for x, y in poly)
            )
            for poly in record["polygons"]
        ]
        per_image.append(
            image_convergence(
                uploaded,
                detail.objects,
                record["width"],
                record["height"],
                label=config.prelabel_class,
            )
        )
    summary = aggregate(per_image)
    summary["round"] = config.round_number
    state = load_state(config.state_path)
    record = state.for_round(config.round_number) or RoundRecord(
        round_number=config.round_number,
        train_project=config.train_project,
        prelabel_project=config.prelabel_project,
        prelabel_class=config.prelabel_class,
    )
    record.convergence = summary
    state.upsert(record)
    save_state(config.state_path, state)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_status(config: Config, args: argparse.Namespace) -> int:
    """Run the ``status`` command: print the per-round trend.

    Parameters
    ----------
    config : Config
        The configuration.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """
    state: LoopState = load_state(config.state_path)
    print(f"rounds recorded: {len(state.rounds)}")
    for record in state.rounds:
        metrics = record.metrics or {}
        print(
            f"  round {record.round_number}: uploaded={record.uploaded} skipped={record.skipped} "
            f"test_iou={metrics.get('test_iou', 'n/a')} "
            f"convergence={record.convergence is not None}"
        )
    trend = [dict(record.convergence or {}) for record in state.rounds if record.convergence]
    print()
    print(trend_table(trend))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        The parser with one subparser per command.
    """
    parser = argparse.ArgumentParser(description="The microsite annotation flywheel.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="configuration file")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="download the latest annotations")
    sync.add_argument("--limit", type=int, default=0, help="maximum annotated images (0 = all)")
    sync.set_defaults(func=cmd_sync)

    train = sub.add_parser("train", help="train the pre-labeller")
    train.add_argument(
        "--random-init", action="store_true", help="train without pretrained weights"
    )
    train.set_defaults(func=cmd_train)

    prelabel = sub.add_parser("prelabel", help="upload editable pre-labels")
    prelabel.add_argument(
        "--checkpoint", default="", help="checkpoint path (default: this round's best)"
    )
    prelabel.add_argument("--limit", type=int, default=0, help="maximum images (0 = all)")
    prelabel.set_defaults(func=cmd_prelabel)

    compare = sub.add_parser("compare", help="measure convergence after correction")
    compare.set_defaults(func=cmd_compare)

    status = sub.add_parser("status", help="show the per-round trend")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the flywheel CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments to parse. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        When the configuration cannot be loaded.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except Exception as exc:  # surface any configuration error to the operator
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status: int = args.func(config, args)
    return status


if __name__ == "__main__":
    sys.exit(main())
