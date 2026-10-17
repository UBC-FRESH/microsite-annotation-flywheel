"""Load and validate the flywheel configuration.

The configuration names the Roboflow workspace and projects, the round number, the paths a
round writes to, and the training and pre-labelling settings. It is deliberately small and
explicit: an operator changes one round by editing ``config.yaml``, not code.

Errors are raised as :class:`ConfigError` with a message that names the offending key, so a
non-coder can fix the file without reading this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME: str = "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class ModelConfig:
    """Training settings for the pre-labeller.

    Attributes
    ----------
    encoder : str
        Encoder backbone name, such as ``resnet34``.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Images per optimisation step.
    learning_rate : float
        Initial AdamW learning rate.
    patience : int
        Early-stopping patience, in epochs, on validation IoU.
    image_size : int
        Square training and inference size in pixels.
    """

    encoder: str = "resnet34"
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 3e-4
    patience: int = 12
    image_size: int = 1024


@dataclass(frozen=True)
class PrelabelConfig:
    """Settings for pre-labelling unannotated images.

    Attributes
    ----------
    threshold : float
        Probability above which a pixel is a predicted microsite.
    min_polygon_area : int
        Connected components smaller than this many pixels are dropped.
    tta : bool
        Whether to average the eight dihedral test-time-augmentation views.
    limit : int
        Maximum number of images to pre-label in one run; ``0`` means all.
    split : str
        Roboflow split assigned to uploaded pre-labels, one of ``train``, ``valid``, ``test``.
    """

    threshold: float = 0.5
    min_polygon_area: int = 64
    tta: bool = True
    limit: int = 0
    split: str = "train"


@dataclass(frozen=True)
class Config:
    """The whole flywheel configuration for one round.

    Attributes
    ----------
    workspace : str
        Roboflow workspace slug.
    train_project : str
        Project whose annotated images feed training.
    prelabel_project : str
        Project whose unannotated images receive pre-labels.
    prelabel_class : str
        Class name written on every uploaded pre-label polygon.
    round_number : int
        Round identifier, stamped into provenance and the loop state.
    data_root : Path
        Directory for round outputs; ignored by git and, in production, a DataLad path.
    state_path : Path
        Path to ``loop_state.json``.
    model : ModelConfig
        Training settings.
    prelabel : PrelabelConfig
        Pre-labelling settings.

    Notes
    -----
    ``data_root`` and ``state_path`` are stored as relative paths in the file and resolved
    against the configuration file's directory, so the repository stays portable.
    """

    workspace: str
    train_project: str
    prelabel_project: str
    prelabel_class: str
    round_number: int
    data_root: Path
    state_path: Path
    model: ModelConfig = ModelConfig()
    prelabel: PrelabelConfig = PrelabelConfig()

    @property
    def round_dir(self) -> Path:
        """Return the directory that holds this round's artifacts."""
        return self.data_root / f"round-{self.round_number}"

    @property
    def dataset_dir(self) -> Path:
        """Return the directory that holds the synced training dataset."""
        return self.round_dir / "dataset"

    @property
    def checkpoints_dir(self) -> Path:
        """Return the directory that holds this round's checkpoints."""
        return self.round_dir / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        """Return the directory that holds this round's predictions and upload records."""
        return self.round_dir / "predictions"


def _as_mapping(value: object, key: str) -> dict[str, Any]:
    """Return ``value`` as a string-keyed mapping, or fail with a clear message.

    Parameters
    ----------
    value : object
        The value read from the YAML document.
    key : str
        The dotted key path, used in the error message.

    Returns
    -------
    dict of str to Any
        The value as a mapping.

    Raises
    ------
    ConfigError
        When the value is not a mapping.
    """
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _require_str(mapping: dict[str, Any], key: str) -> str:
    """Return a required non-empty string value from a mapping.

    Parameters
    ----------
    mapping : dict of str to Any
        The mapping to read.
    key : str
        The key to read.

    Returns
    -------
    str
        The stripped string value.

    Raises
    ------
    ConfigError
        When the key is missing, not a string, or empty.
    """
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _require_int(mapping: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    """Return a required integer value from a mapping.

    Parameters
    ----------
    mapping : dict of str to Any
        The mapping to read.
    key : str
        The key to read.
    minimum : int, optional
        Smallest accepted value.

    Returns
    -------
    int
        The integer value.

    Raises
    ------
    ConfigError
        When the key is missing, not an integer, a boolean, or below ``minimum``.
    """
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer")
    if value < minimum:
        raise ConfigError(f"'{key}' must be at least {minimum}")
    return value


def _optional_float(mapping: dict[str, Any], key: str, default: float) -> float:
    """Return a float value from a mapping, or a default.

    Parameters
    ----------
    mapping : dict of str to Any
        The mapping to read.
    key : str
        The key to read.
    default : float
        Value used when the key is absent.

    Returns
    -------
    float
        The float value or the default.

    Raises
    ------
    ConfigError
        When the key is present but not a number.
    """
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number")
    return float(value)


def _optional_bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    """Return a boolean value from a mapping, or a default.

    Parameters
    ----------
    mapping : dict of str to Any
        The mapping to read.
    key : str
        The key to read.
    default : bool
        Value used when the key is absent.

    Returns
    -------
    bool
        The boolean value or the default.

    Raises
    ------
    ConfigError
        When the key is present but not a boolean.
    """
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be true or false")
    return value


def parse_config(document: dict[str, Any], *, base_dir: Path) -> Config:
    """Build a :class:`Config` from a parsed YAML document.

    Parameters
    ----------
    document : dict of str to Any
        The YAML document, as returned by ``yaml.safe_load``.
    base_dir : Path
        Directory that relative paths resolve against, normally the config file's parent.

    Returns
    -------
    Config
        The validated configuration.

    Raises
    ------
    ConfigError
        When a required key is missing or any value is invalid.
    """
    top = _as_mapping(document, "<root>")
    round_number = _require_int(top, "round", minimum=1)
    train_project = _require_str(top, "train_project")
    prelabel_project = _require_str(top, "prelabel_project")
    workspace = _require_str(top, "workspace")
    prelabel_class = _require_str(top, "prelabel_class")

    model_map = _as_mapping(top.get("model", {}), "model")
    model = ModelConfig(
        encoder=_require_str(model_map, "encoder"),
        epochs=_require_int(model_map, "epochs", minimum=1),
        batch_size=_require_int(model_map, "batch_size", minimum=1),
        learning_rate=_optional_float(model_map, "learning_rate", 3e-4),
        patience=_require_int(model_map, "patience", minimum=1),
        image_size=_require_int(model_map, "image_size", minimum=32),
    )

    prelabel_raw = _as_mapping(top.get("prelabel", {}), "prelabel")
    split = prelabel_raw.get("split", "train")
    if split not in ("train", "valid", "test"):
        raise ConfigError("'prelabel.split' must be one of train, valid, test")
    prelabel = PrelabelConfig(
        threshold=_optional_float(prelabel_raw, "threshold", 0.5),
        min_polygon_area=_require_int(prelabel_raw, "min_polygon_area", minimum=0)
        if "min_polygon_area" in prelabel_raw
        else 64,
        tta=_optional_bool(prelabel_raw, "tta", True),
        limit=_require_int(prelabel_raw, "limit", minimum=0) if "limit" in prelabel_raw else 0,
        split=split,
    )

    data_root_raw = top.get("data_root", "_data")
    if not isinstance(data_root_raw, str) or not data_root_raw.strip():
        raise ConfigError("'data_root' must be a non-empty string")
    state_raw = top.get("state_path", "loop_state.json")
    if not isinstance(state_raw, str) or not state_raw.strip():
        raise ConfigError("'state_path' must be a non-empty string")

    return Config(
        workspace=workspace,
        train_project=train_project,
        prelabel_project=prelabel_project,
        prelabel_class=prelabel_class,
        round_number=round_number,
        data_root=(base_dir / data_root_raw.strip()).resolve(),
        state_path=(base_dir / state_raw.strip()).resolve(),
        model=model,
        prelabel=prelabel,
    )


def load_config(path: Path) -> Config:
    """Read and validate a configuration file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
        The validated configuration.

    Raises
    ------
    ConfigError
        When the file is missing, is not valid YAML, or fails validation.
    """
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"configuration file is not valid YAML: {exc}") from exc
    if document is None:
        raise ConfigError(f"configuration file is empty: {path}")
    return parse_config(_as_mapping(document, "<root>"), base_dir=path.resolve().parent)
