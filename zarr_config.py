"""Parses the zarr output specification (the `variables` and `encoding` sections
of reforecast_config.yaml).

Defines *what* gets stored and *how*: which model variables are saved, the
units they carry, how each is aggregated in time, and the on-disk encoding.
Anything omitted from the config's `encoding` section falls back to the
defaults in this module, which are the reference archive's layout.
"""

import dataclasses
from collections.abc import Sequence

import yaml

# Aggregation identifiers as they appear in the config. zarr_stream's reducer
# table is keyed by these constants.
NATIVE = "native"
DAILY_MEAN = "daily_mean"
DAILY_MIN = "daily_min"
DAILY_MAX = "daily_max"
DAILY_SUM = "daily_sum"
DAILY_AGGREGATIONS = (DAILY_MEAN, DAILY_MIN, DAILY_MAX, DAILY_SUM)
AGGREGATIONS = (NATIVE,) + DAILY_AGGREGATIONS

# `native` never takes a suffix; daily aggregations do when a variable lists
# more than one aggregation.
_SUFFIXES = {
    DAILY_MEAN: "_mean",
    DAILY_MIN: "_min",
    DAILY_MAX: "_max",
    DAILY_SUM: "_sum",
}

# Short (ECMWF-style) names -> WeatherNext2 surface variables.
SURFACE_SOURCES = {
    "2t": "2m_temperature",
    "sst": "sea_surface_temperature",
    "msl": "mean_sea_level_pressure",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "tp": "total_precipitation_6hr",
}

# Short name bases -> WeatherNext2 pressure-level variables; configured as
# `<base>_<level>`, e.g. `z_500`.
LEVEL_SOURCES = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}

# Reference archive layout: on-disk encoding defaults.
DEFAULT_CHUNKS = {
    "time": 1,
    "number": 1,
    "prediction_timedelta": 24,
    "prediction_timedelta_daily": 10,
    "lat": 90,
    "lon": 180,
}
DEFAULT_SHARDS = {
    "time": 1,
    "number": 1,
    "prediction_timedelta": 168,
    "prediction_timedelta_daily": 50,
    "lat": 720,
    "lon": 1440,
}
DEFAULT_COMPRESSOR = {"cname": "zstd", "clevel": 7, "shuffle": "bitshuffle"}
DEFAULT_KEEPBITS = None


@dataclasses.dataclass(frozen=True)
class OutputVariable:
    """One array in the output store."""

    name: str  # Output name, e.g. "2t" or "2t_min".
    short_name: str  # Configured name, e.g. "2t".
    model_var: str  # WeatherNext2 variable it is derived from.
    level: int | None  # Pressure level (hPa), None for surface variables.
    units: str
    aggregation: str  # One of AGGREGATIONS.

    @property
    def is_native(self) -> bool:
        return self.aggregation == NATIVE


@dataclasses.dataclass(frozen=True)
class EncodingSpec:
    chunks: dict
    shards: dict
    compressor: dict
    keepbits: int | None
    keepbits_by_variable: dict

    def keepbits_for(self, name: str) -> int | None:
        return self.keepbits_by_variable.get(name, self.keepbits)


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    outputs: tuple[OutputVariable, ...]
    encoding: EncodingSpec

    @property
    def native_outputs(self) -> tuple[OutputVariable, ...]:
        return tuple(o for o in self.outputs if o.is_native)

    @property
    def daily_outputs(self) -> tuple[OutputVariable, ...]:
        return tuple(o for o in self.outputs if not o.is_native)

    @property
    def model_variables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.model_var for o in self.outputs))

    def uncompressed_bytes(
        self,
        *,
        n_members: int | None,
        n_steps: int,
        n_daily: int,
        n_lat: int,
        n_lon: int,
    ) -> int:
        """Size of one store's float32 arrays before compression.

        The on-disk size is this times the compression ratio, which depends on
        the field and on `encoding.keepbits`; see submit_reforecast.py.
        """
        members = n_members or 1
        per_step = members * n_lat * n_lon * 4
        return per_step * (
            len(self.native_outputs) * n_steps + len(self.daily_outputs) * n_daily
        )


def _resolve_source(short_name: str) -> tuple[str, int | None]:
    """Maps a configured short name to (model variable, pressure level)."""
    if short_name in SURFACE_SOURCES:
        return SURFACE_SOURCES[short_name], None
    base, _, level_str = short_name.rpartition("_")
    if base in LEVEL_SOURCES and level_str.isdigit():
        return LEVEL_SOURCES[base], int(level_str)
    raise ValueError(
        f"Unknown variable '{short_name}': not a surface short name "
        f"({sorted(SURFACE_SOURCES)}) nor '<base>_<level>' with base in "
        f"{sorted(LEVEL_SOURCES)}."
    )


def _parse_variables(
    variables: dict, available_levels: Sequence[int] | None
) -> tuple[OutputVariable, ...]:
    """Expands the config's variable table into per-aggregation outputs."""
    outputs = []
    for short_name, entry in variables.items():
        short_name = str(short_name)
        if not isinstance(entry, dict) or "units" not in entry:
            raise ValueError(
                f"Variable '{short_name}' must be a mapping with "
                "'units' and 'aggregations'."
            )
        aggregations = entry.get("aggregations")
        if not aggregations or not isinstance(aggregations, list):
            raise ValueError(
                f"Variable '{short_name}' needs a non-empty 'aggregations' list."
            )
        if len(set(aggregations)) != len(aggregations):
            raise ValueError(f"Variable '{short_name}' repeats an aggregation.")
        for aggregation in aggregations:
            if aggregation not in AGGREGATIONS:
                raise ValueError(
                    f"Variable '{short_name}': unknown aggregation '{aggregation}' "
                    f"(expected one of {AGGREGATIONS})."
                )
        unknown_keys = set(entry) - {"units", "aggregations"}
        if unknown_keys:
            raise ValueError(
                f"Variable '{short_name}': unknown keys {sorted(unknown_keys)}."
            )

        model_var, level = _resolve_source(short_name)
        if (
            level is not None
            and available_levels is not None
            and level not in available_levels
        ):
            raise ValueError(
                f"Variable '{short_name}': level {level} not among the model's "
                f"levels {sorted(available_levels)}."
            )

        # A single aggregation keeps the bare name; several get one output per
        # aggregation, suffixed except for `native`.
        for aggregation in aggregations:
            if len(aggregations) == 1 or aggregation == NATIVE:
                name = short_name
            else:
                name = short_name + _SUFFIXES[aggregation]
            outputs.append(
                OutputVariable(
                    name=name,
                    short_name=short_name,
                    model_var=model_var,
                    level=level,
                    units=str(entry["units"]),
                    aggregation=aggregation,
                )
            )

    names = [o.name for o in outputs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate output names: {sorted(duplicates)}.")
    return tuple(outputs)


def _parse_encoding(encoding: dict) -> EncodingSpec:
    unknown_keys = set(encoding) - {
        "chunks",
        "shards",
        "compressor",
        "keepbits",
        "keepbits_by_variable",
    }
    if unknown_keys:
        raise ValueError(f"Unknown encoding keys {sorted(unknown_keys)}.")

    chunks = {**DEFAULT_CHUNKS, **(encoding.get("chunks") or {})}
    shards = {**DEFAULT_SHARDS, **(encoding.get("shards") or {})}
    for dim in set(chunks) | set(shards):
        if dim not in DEFAULT_CHUNKS:
            raise ValueError(f"Unknown chunk/shard dimension '{dim}'.")
        if chunks[dim] < 1:
            raise ValueError(f"Chunk size for '{dim}' must be >= 1.")
        if shards[dim] % chunks[dim] != 0:
            raise ValueError(
                f"Shard size for '{dim}' ({shards[dim]}) must be a positive "
                f"multiple of the chunk size ({chunks[dim]})."
            )

    keepbits = encoding.get("keepbits", DEFAULT_KEEPBITS)
    keepbits_by_variable = encoding.get("keepbits_by_variable") or {}
    for name, bits in {**keepbits_by_variable, "keepbits": keepbits}.items():
        if bits is not None and not (1 <= int(bits) <= 23):
            raise ValueError(f"keepbits for '{name}' must be in [1, 23] or null.")

    return EncodingSpec(
        chunks=chunks,
        shards=shards,
        compressor={**DEFAULT_COMPRESSOR, **(encoding.get("compressor") or {})},
        keepbits=keepbits,
        keepbits_by_variable=dict(keepbits_by_variable),
    )


def parse(
    variables: dict,
    encoding: dict | None = None,
    available_levels: Sequence[int] | None = None,
) -> OutputSpec:
    """Validates the `variables`/`encoding` sections of an already-loaded config.

    Split out from `load` so a larger config that embeds these two sections —
    reforecast_config.yaml — parses them the same way.
    """
    if not variables:
        raise ValueError("The config defines no variables.")
    return OutputSpec(
        outputs=_parse_variables(variables, available_levels),
        encoding=_parse_encoding(encoding or {}),
    )


def load(path: str, available_levels: Sequence[int] | None = None) -> OutputSpec:
    """Loads and validates an output spec from a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown_keys = set(raw) - {"variables", "encoding"}
    if unknown_keys:
        raise ValueError(f"Unknown top-level config keys {sorted(unknown_keys)}.")
    if not raw.get("variables"):
        raise ValueError(f"Config {path} defines no variables.")
    return parse(raw["variables"], raw.get("encoding"), available_levels)
