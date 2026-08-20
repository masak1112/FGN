"""Parses reforecast_config.yaml, the reforecast campaign's single source of truth.

The file carries the `variables` and `encoding` sections that zarr_config
already understands, plus three campaign sections:

  resources   Slurm batch geometry. Read by submit_reforecast.py.
  reforecast  What to run: ensemble size, forecast length, which years.
  data        Where published stores land, and with which permissions.

Every key is optional; anything omitted falls back to the defaults in this
module. Unknown keys are rejected rather than ignored, so a typo in a config
fails at submit time instead of quietly changing what runs.
"""

import dataclasses
import os
import pathlib
import re
from collections.abc import Sequence

import pandas as pd
import yaml

import init_dates
import zarr_config

# Checkpoints (independent training seeds) that ensemble members are pooled
# from. `n_members / members_per_checkpoint` must equal how many are listed.
DEFAULT_WEIGHTS = tuple(
    f"data/params/WeatherNext2_lt2025_model{i}.npz" for i in (1, 2, 3, 4)
)
DEFAULT_CONFIG_NAME = "weathernext2/configs/WeatherNext2"

# The model's native output grid, used only to estimate store sizes.
LAT_SIZE, LON_SIZE = 721, 1440
TIMESTEP_HOURS = 6

# Reforecast stores are named like the reference archive in
# /net/monsoon/reforecast: one store per init, `<YYYY-MM-DD>T<HH>.zarr`.
STORE_NAME_FORMAT = "%Y-%m-%dT%H"

# Directories 2770 and files 660: group-shared, not world-readable. The setgid
# bit keeps the group on everything created underneath.
GROUP_UMASK = 0o007
GROUP_DIR_MODE = 0o2770


def _parse_section(cls, raw: dict | None, section: str):
    """Builds one config dataclass, rejecting keys it does not define."""
    raw = dict(raw or {})
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in the '{section}' section "
            f"(known: {sorted(fields)})."
        )
    return cls(**raw)


@dataclasses.dataclass(frozen=True)
class Resources:
    """Slurm geometry for one array task, i.e. for one job."""

    n_jobs: int = 1
    # Array throttle: at most this many tasks run at once. None means no limit.
    n_concurrent_jobs: int | None = None
    nodes_per_job: int = 1
    gpu_per_job: int = 4
    gpu_type: str | None = None
    cpu_per_task: int = 16
    mem_per_job: str = "256GB"
    partition: str = "general"
    time_limit: str = "12:00:00"
    exclude_nodes: str | None = None
    log_dir: str = "./logs/reforecast"
    job_name: str = "fgn_reforecast"

    def __post_init__(self):
        for name in ("n_jobs", "nodes_per_job", "gpu_per_job", "cpu_per_task"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"resources.{name} must be >= 1.")
        if self.n_concurrent_jobs is not None and int(self.n_concurrent_jobs) < 1:
            raise ValueError("resources.n_concurrent_jobs must be >= 1 or null.")
        self.time_limit_seconds  # Validated by parsing.

    @property
    def gres(self) -> str:
        """The --gres value, with the GPU type pinned when one is configured."""
        if self.gpu_type:
            return f"gpu:{self.gpu_type}:{self.gpu_per_job}"
        return f"gpu:{self.gpu_per_job}"

    @property
    def time_limit_seconds(self) -> int:
        """Parses the Slurm --time formats into seconds.

        Slurm accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm and d-hh:mm:ss: a
        `d-` prefix shifts the leading field from minutes/hours to hours.
        """
        if not isinstance(self.time_limit, str):
            # YAML 1.1 reads an unquoted 12:00:00 as sexagesimal, i.e. the int
            # 43200, which is indistinguishable from a bare minute count.
            raise ValueError(
                f"resources.time_limit must be quoted in the YAML: write "
                f'time_limit: "12:00:00", not 12:00:00 (which parsed as '
                f"{self.time_limit!r})."
            )
        text = self.time_limit.strip()
        if not re.fullmatch(r"(\d+-)?\d+(:\d+){0,2}", text):
            raise ValueError(
                f"resources.time_limit '{text}' is not a Slurm time "
                "(mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm or d-hh:mm:ss)."
            )
        days, _, rest = text.rpartition("-")
        parts = [int(p) for p in rest.split(":")]
        if days:  # d-hh[:mm[:ss]]
            hours, minutes, seconds = (parts + [0, 0])[:3]
        elif len(parts) == 3:  # hh:mm:ss
            hours, minutes, seconds = parts
        elif len(parts) == 2:  # mm:ss
            hours, minutes, seconds = 0, parts[0], parts[1]
        else:  # Bare minutes.
            hours, minutes, seconds = 0, parts[0], 0
        return ((int(days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds


@dataclasses.dataclass(frozen=True)
class Reforecast:
    """What the campaign produces."""

    n_members: int = 32
    members_per_checkpoint: int = 8
    lead_hours: int = 1200
    year_range: tuple[int, int] = (2000, 2025)  # Inclusive.
    # Calendar months to leave out entirely, e.g. [6, 7, 8, 9] for non-monsoon.
    exclude_months: tuple[int, ...] = ()
    weights: tuple[str, ...] = DEFAULT_WEIGHTS
    config_name: str = DEFAULT_CONFIG_NAME
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "year_range", tuple(self.year_range))
        object.__setattr__(self, "exclude_months", tuple(self.exclude_months or ()))
        object.__setattr__(self, "weights", tuple(self.weights))
        if len(self.year_range) != 2 or self.year_range[0] > self.year_range[1]:
            raise ValueError(
                f"reforecast.year_range {self.year_range} must be "
                "[first_year, last_year] with first <= last."
            )
        if self.lead_hours < TIMESTEP_HOURS or self.lead_hours % TIMESTEP_HOURS:
            raise ValueError(
                f"reforecast.lead_hours ({self.lead_hours}) must be a positive "
                f"multiple of {TIMESTEP_HOURS}."
            )
        if self.n_members < 1 or self.members_per_checkpoint < 1:
            raise ValueError("reforecast member counts must be >= 1.")
        if self.n_members % self.members_per_checkpoint:
            raise ValueError(
                f"reforecast.n_members ({self.n_members}) must be a multiple of "
                f"members_per_checkpoint ({self.members_per_checkpoint})."
            )
        if self.n_checkpoints != len(self.weights):
            raise ValueError(
                f"reforecast.n_members / members_per_checkpoint = "
                f"{self.n_checkpoints} checkpoints, but {len(self.weights)} "
                "weights file(s) are configured."
            )
        for month in self.exclude_months:
            if not 1 <= int(month) <= 12:
                raise ValueError(
                    f"reforecast.exclude_months has an invalid month {month!r}."
                )

    @property
    def n_checkpoints(self) -> int:
        return self.n_members // self.members_per_checkpoint

    @property
    def n_steps(self) -> int:
        return self.lead_hours // TIMESTEP_HOURS

    @property
    def n_daily(self) -> int:
        """Complete UTC days in the forecast, i.e. the daily-aggregate length.

        Mirrors zarr_stream's windowing: a step valid at t covers (t-6h, t], so
        a 00Z init's first day owns the steps valid at 06/12/18/24h and whole
        days divide the lead exactly.
        """
        return self.lead_hours // 24


@dataclasses.dataclass(frozen=True)
class Data:
    """Where the published stores go."""

    data_dir: str = "forecasts"
    group_permissions: bool = True


@dataclasses.dataclass(frozen=True)
class ReforecastConfig:
    path: str
    resources: Resources
    reforecast: Reforecast
    data: Data
    output_spec: zarr_config.OutputSpec

    def init_times(self) -> list[pd.Timestamp]:
        """Every init the campaign covers, ascending.

        The per-year calendar comes from init_dates.get_dates (00Z, roughly
        every 6 days plus the 9th and 17th, ~91 per year).
        """
        first, last = self.reforecast.year_range
        dates: list = []
        for year in range(first, last + 1):
            dates.extend(init_dates.get_dates(year))
        if self.reforecast.exclude_months:
            dates = init_dates.filter_months(
                dates, list(self.reforecast.exclude_months)
            )
        return [pd.Timestamp(d) for d in sorted(dates)]

    def store_path(self, init: pd.Timestamp, extension: str = "zarr") -> pathlib.Path:
        """Published path for one init, e.g. <data_dir>/2000-01-01T00.zarr."""
        name = pd.Timestamp(init).strftime(STORE_NAME_FORMAT)
        return pathlib.Path(self.data.data_dir) / f"{name}.{extension}"

    def staging_path(self, init: pd.Timestamp, extension: str = "zarr") -> pathlib.Path:
        """The `_partial` path zarr_stream stages into before publishing."""
        path = self.store_path(init, extension)
        return path.with_name(path.stem + "_partial" + path.suffix)

    def store_bytes(self) -> int:
        """Uncompressed size of one published store."""
        return self.output_spec.uncompressed_bytes(
            n_members=self.reforecast.n_members,
            n_steps=self.reforecast.n_steps,
            n_daily=self.reforecast.n_daily,
            n_lat=LAT_SIZE,
            n_lon=LON_SIZE,
        )


def load(path: str, available_levels: Sequence[int] | None = None) -> ReforecastConfig:
    """Loads and validates a reforecast config.

    `available_levels` is checked against the model's pressure levels when
    given; pass None to parse the config without importing the model.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"resources", "reforecast", "data", "variables", "encoding"}
    if unknown:
        raise ValueError(f"Unknown top-level keys {sorted(unknown)} in {path}.")

    resources = _parse_section(Resources, raw.get("resources"), "resources")
    reforecast = _parse_section(Reforecast, raw.get("reforecast"), "reforecast")
    data = _parse_section(Data, raw.get("data"), "data")
    if reforecast.members_per_checkpoint % resources.gpu_per_job:
        raise ValueError(
            f"reforecast.members_per_checkpoint "
            f"({reforecast.members_per_checkpoint}) must be a multiple of "
            f"resources.gpu_per_job ({resources.gpu_per_job}): members run in "
            "parallel across the job's GPUs via pmap."
        )
    return ReforecastConfig(
        path=os.fspath(path),
        resources=resources,
        reforecast=reforecast,
        data=data,
        output_spec=zarr_config.parse(
            raw.get("variables"), raw.get("encoding"), available_levels
        ),
    )


def apply_group_permissions() -> None:
    """Makes everything this process creates group-writable, not world-readable.

    Files land 0660 and directories 0770. Call this before anything is written:
    the umask cannot be applied retroactively. Slurm propagates the submitting
    process's umask to the batch job, so calling this in the submitter is also
    what makes the job's own stdout file group-writable — the `umask` in the
    batch script only covers files the job creates after it starts.

    Group *ownership* is a separate mechanism: new files take the creating
    process's primary group (here `marchakitus`) unless the parent directory is
    setgid, which every directory under /net/monsoon is. GROUP_DIR_MODE keeps
    that bit on directories this code creates so the inheritance continues.
    """
    os.umask(GROUP_UMASK)


def make_dir(path, group_permissions: bool = True) -> pathlib.Path:
    """Creates a directory, setgid when sharing with the group.

    Every directory created here gets GROUP_DIR_MODE, not just the leaf:
    mkdir(parents=True) applies only the umask to the intermediates, which
    leaves them group-read-only whenever the ambient umask is not 007.
    Pre-existing directories keep their mode, except the leaf, which is the one
    the caller explicitly asked to be group-shared.
    """
    path = pathlib.Path(path)
    created = (
        [p for p in (path, *path.parents) if not p.exists()]
        if group_permissions
        else []
    )
    path.mkdir(parents=True, exist_ok=True)
    for directory in {*created, path} if group_permissions else ():
        try:
            directory.chmod(GROUP_DIR_MODE)
        except OSError:
            pass  # Pre-existing directory owned by someone else in the group.
    return path
