"""Run WeatherNext 2 (FGN) inference from ERA5 initial conditions.

Sources initial conditions from the ARCO ERA5 zarr store, runs an
autoregressive rollout with a pretrained WeatherNext 2 checkpoint, and streams
the forecast to a zarr store (or saves a NetCDF file).

Ensemble size, forecast length, output location and encoding all come from
reforecast_config.yaml; command-line flags override individual values. Several
init dates can run in one process, which is how the reforecast campaign is
executed: the model compiles once (~3 min) and every date after the first
reuses that compilation.

Examples:
  # One date, everything else from the config.
  uv run python wn2_inference.py --init 20260625T00

  # A campaign slice: every init in the config's year range that has no
  # published store yet.
  uv run python wn2_inference.py --all_pending

  # The date list one array task was handed by submit_reforecast.py.
  uv run python wn2_inference.py --init_file logs/reforecast/<run>/dates/task_000.txt
"""

import argparse
import contextlib
import dataclasses
import functools
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time as time_module

import dask.array
import haiku as hk
import jax
import numpy as np
import pandas as pd
import psutil
import xarray
import xarray_jax

import reforecast_config
import zarr_stream
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn

ERA5_PATH ="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_CONFIG_FILE = "reforecast_config.yaml"
TIMESTEP = pd.Timedelta("6h")

# Variables in the ERA5 store that WN2 takes as inputs.
DYNAMIC_3D_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
DYNAMIC_2D_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "sea_surface_temperature",
)
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")

logger = logging.getLogger("wn2")
perf_logger = logging.getLogger("wn2.perf")
resource_logger = logging.getLogger("wn2.resources")


_FORMATTER = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
)

# `wn2.log` receives everything via propagation from the child loggers; the
# perf and resource streams additionally get their own files.
_LOG_FILES = (
    (logger, "wn2.log"),
    (perf_logger, "perf.log"),
    (resource_logger, "resources.log"),
)


def setup_logging() -> None:
    """Sends the `wn2` loggers to the console."""
    console = logging.StreamHandler()
    console.setFormatter(_FORMATTER)
    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    # JAX/absl may configure the root logger; don't let records bubble up to it
    # or every message prints twice.
    logger.propagate = False


@contextlib.contextmanager
def log_files(log_dir: str | os.PathLike | None, group_permissions: bool = False):
    """Tees the `wn2` loggers into files under `log_dir` for the block's duration.

    Used twice over: once around the whole run, and again per init date, so
    every date gets a self-contained wn2/perf/resources trio while the run-level
    files keep the full picture.
    """
    if not log_dir:
        yield None
        return
    log_dir = reforecast_config.make_dir(log_dir, group_permissions)
    attached = []
    try:
        for lg, filename in _LOG_FILES:
            handler = logging.FileHandler(log_dir / filename)
            handler.setFormatter(_FORMATTER)
            lg.addHandler(handler)
            attached.append((lg, handler))
        yield log_dir
    finally:
        for lg, handler in attached:
            lg.removeHandler(handler)
            handler.close()


@contextlib.contextmanager
def timed(stage: str):
    """Logs the wall-clock duration of a stage to the performance logger."""
    start = time_module.perf_counter()
    try:
        yield
    finally:
        perf_logger.info("%s: %.2fs", stage, time_module.perf_counter() - start)


_PROCESS = psutil.Process()


def resource_snapshot() -> str:
    """One-line summary of process CPU/RAM, host RAM, and accelerator memory."""
    host = psutil.virtual_memory()
    # cpu_percent measures since the previous call on the same Process object.
    parts = [
        f"cpu={_PROCESS.cpu_percent():.0f}%",
        f"rss={_PROCESS.memory_info().rss / 2**30:.1f}GiB",
        f"host_mem={host.percent:.0f}%",
    ]
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        if "bytes_in_use" in stats:
            used = stats["bytes_in_use"] / 2**30
            peak = stats.get("peak_bytes_in_use", 0) / 2**30
            parts.append(
                f"{device.platform}{device.id}_mem={used:.1f}GiB(peak={peak:.1f}GiB)"
            )
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
            for i, line in enumerate(smi.strip().splitlines()):
                util, mem = (field.strip() for field in line.split(","))
                parts.append(
                    f"gpu{i}_util={util}% gpu{i}_smi_mem={int(mem) / 1024:.1f}GiB"
                )
        except (subprocess.SubprocessError, ValueError):
            pass
    return " ".join(parts)


class ResourceMonitor:
    """Periodically logs resource utilization to the `wn2.resources` logger."""

    def __init__(self, interval_seconds: float):
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="resource-monitor", daemon=True
        )

    def start(self) -> None:
        _PROCESS.cpu_percent()  # Prime the CPU counter.
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        resource_logger.info("final: %s", resource_snapshot())

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            resource_logger.info(resource_snapshot())


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Reforecast config: ensemble size, forecast length, init dates, "
        "output location and zarr encoding. Default: %(default)s.",
    )

    dates = p.add_argument_group("init dates (choose one; default --all_pending)")
    dates.add_argument(
        "--init",
        nargs="+",
        default=None,
        metavar="INIT",
        help="One or more forecast init times, e.g. 20260625T00 2026-06-26T00.",
    )
    dates.add_argument(
        "--init_file",
        default=None,
        help="File of init times, one per line; blank lines and #comments are "
        "ignored. Written per array task by submit_reforecast.py.",
    )
    dates.add_argument(
        "--all_pending",
        action="store_true",
        help="Every init in the config's year_range without a published store.",
    )
    dates.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip inits whose store already exists instead of failing, so a "
        "job can be resubmitted safely. Default: skip.",
    )

    overrides = p.add_argument_group("config overrides")
    overrides.add_argument(
        "--lead_hours",
        type=int,
        default=None,
        help="Forecast length in hours; must be a multiple of 6.",
    )
    overrides.add_argument(
        "--ensemble_size",
        type=int,
        default=None,
        help="Total number of ensemble members; must be a multiple of the "
        "number of --weights files. Members are split evenly across "
        "checkpoints (FGN multi-seed ensembling).",
    )
    overrides.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="One or more WeatherNext2 .npz checkpoints (independent training "
        "seeds). Ensemble members are pooled equally from each.",
    )
    overrides.add_argument(
        "--config_name",
        default=None,
        help="Fiddle config name within the weathernext package.",
    )
    overrides.add_argument(
        "--data_dir",
        default=None,
        help="Directory the per-init stores are written to.",
    )
    overrides.add_argument("--seed", type=int, default=None, help="Base RNG seed.")

    output = p.add_argument_group("output")
    output.add_argument(
        "--output",
        default=None,
        help="Explicit output path, for a single init only. Default: "
        "<data_dir>/<YYYY-MM-DD>T<HH>.zarr|.nc depending on --format.",
    )
    output.add_argument(
        "--format",
        choices=("zarr", "netcdf"),
        default="zarr",
        help="zarr: stream aggregated outputs per the config's variables "
        "during the rollout (bounded memory). netcdf: accumulate everything "
        "in RAM and save all model variables at the end.",
    )
    output.add_argument(
        "--save_cyclone_vars",
        action="store_true",
        help="Also save the discretized cyclone head outputs "
        "(mostly-NaN sparse fields). --format netcdf only.",
    )
    output.add_argument(
        "--era5_path", default=ERA5_PATH, help="Path to the ARCO ERA5 zarr store."
    )
    output.add_argument(
        "--log_dir",
        default=None,
        help="Directory for wn2.log, perf.log and resources.log, plus a "
        "per-init subdirectory for each. Default: the config's "
        "resources.log_dir; pass '' to log to the console only.",
    )
    output.add_argument(
        "--resource_log_interval",
        type=float,
        default=60.0,
        help="Seconds between resource utilization log samples.",
    )
    return p.parse_args()


def resolve_init_times(args, cfg) -> list[pd.Timestamp]:
    """Determines which inits to run, from the flags or the config's calendar."""
    selectors = [bool(args.init), bool(args.init_file), args.all_pending]
    if sum(selectors) > 1:
        raise ValueError("Pass only one of --init, --init_file and --all_pending.")

    if args.init:
        inits = [pd.Timestamp(value) for value in args.init]
    elif args.init_file:
        text = pathlib.Path(args.init_file).read_text()
        lines = [line.partition("#")[0].strip() for line in text.splitlines()]
        inits = [pd.Timestamp(line) for line in lines if line]
        if not inits:
            raise ValueError(f"{args.init_file} lists no init times.")
    else:
        inits = cfg.init_times()
        logger.info(
            "Init dates from %s: %d date(s) over %d-%d.",
            cfg.path,
            len(inits),
            *cfg.reforecast.year_range,
        )
    return sorted(dict.fromkeys(inits))


def load_era5_initial_conditions(
    era5_path: str, init: pd.Timestamp, levels: tuple[int, ...]
) -> xarray.Dataset:
    """Loads the two WN2 input frames (init - 6h, init) from ARCO ERA5."""
    era5 = xarray.open_zarr(era5_path, chunks=None, storage_options={"token": "anon"})

    final_stop = pd.Timestamp(era5.attrs.get("valid_time_stop", "2100"))
    era5t_stop = pd.Timestamp(
        era5.attrs.get(
            "valid_time_stop_era5t", era5.attrs.get("valid_time_stop", "2100")
        )
    ) + pd.Timedelta("23h")
    if init > era5t_stop:
        raise ValueError(
            f"Init time {init} is beyond ERA5 data availability "
            f"({era5t_stop}) in {era5_path}."
        )
    if init > final_stop:
        logger.warning(
            "Init %s is past the final ERA5 cutoff (%s); using preliminary ERA5T data.",
            init,
            final_stop,
        )

    input_times = [init - TIMESTEP, init]
    frames = era5[list(DYNAMIC_3D_VARS + DYNAMIC_2D_VARS)].sel(
        time=input_times, level=list(levels)
    )
    statics = era5[list(STATIC_VARS)].sel(time=init)

    logger.info("Loading ERA5 initial conditions for %s ...", input_times)
    with timed("era5_input_load"):
        frames = frames.load()
        statics = statics.load()

    ds = xarray.merge([frames, statics.drop_vars("time")])
    # Match the WN2 batch conventions: lat ascending, names lat/lon, float32,
    # a leading batch dim, and a relative "time" coord with "datetime" attached.
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sortby("lat")
    ds = ds.astype("float32")
    ds = ds.assign_coords(
        lat=ds.lat.astype("float32"),
        lon=ds.lon.astype("float32"),
        level=ds.level.astype("int32"),
    )
    dynamic = ds[list(DYNAMIC_3D_VARS + DYNAMIC_2D_VARS)].expand_dims("batch", axis=0)
    dynamic = dynamic.assign_coords(
        datetime=(("batch", "time"), pd.DatetimeIndex(input_times).values[None, :]),
        time=(np.arange(2) * TIMESTEP.to_numpy()),
    )
    ds = xarray.merge(
        [dynamic, ds[list(STATIC_VARS)].drop_vars("datetime", errors="ignore")]
    )

    for var in ds.data_vars:
        nan_frac = float(np.isnan(ds[var]).mean())
        if nan_frac > 0 and var != "sea_surface_temperature":
            raise ValueError(f"ERA5 input '{var}' has {nan_frac:.1%} NaNs.")
    return ds


def build_batch(
    inputs: xarray.Dataset,
    task,
    num_steps: int,
) -> xarray.Dataset:
    """Appends lazy NaN target frames to the 2 input frames."""
    target_only_vars = tuple(
        v for v in task.target_variables if v not in inputs.data_vars
    )

    def nan_frames(template: xarray.Dataset, var: str, times, datetimes):
        like = template["2m_temperature"]
        shape = (1, len(times)) + like.shape[2:]
        return xarray.DataArray(
            dask.array.full(shape, np.nan, chunks=(1, 1, -1, -1), dtype=np.float32),
            dims=("batch", "time", "lat", "lon"),
            coords={
                "time": times,
                "lat": like.lat,
                "lon": like.lon,
                "datetime": (("batch", "time"), datetimes[None, :]),
            },
            name=var,
        )

    input_times = inputs["time"].values
    input_datetimes = inputs["datetime"].isel(batch=0).values
    # Target-only variables (precip + cyclone heads) are never read at input
    # times, but must exist there for the time concat to be well-formed.
    for var in target_only_vars:
        inputs[var] = nan_frames(inputs, var, input_times, input_datetimes)

    target_times = input_times[-1] + TIMESTEP.to_numpy() * np.arange(1, num_steps + 1)
    target_datetimes = input_datetimes[-1] + (target_times - input_times[-1])

    future_vars = {}
    for var in task.target_variables:
        like = inputs[var]
        shape = list(like.shape)
        shape[like.dims.index("time")] = num_steps
        chunks = tuple(1 if d == "time" else -1 for d in like.dims)
        future_vars[var] = xarray.DataArray(
            dask.array.full(tuple(shape), np.nan, chunks=chunks, dtype=np.float32),
            dims=like.dims,
            coords={
                k: v
                for k, v in like.coords.items()
                if k != "datetime" and "time" not in v.dims
            },
            name=var,
        )
    future = xarray.Dataset(future_vars).assign_coords(
        time=target_times,
        datetime=(("batch", "time"), target_datetimes[None, :]),
    )
    return xarray.concat([inputs, future], dim="time", data_vars="minimal")


def build_predictor_fn(config):
    """Returns a pmapped predictor taking (replicated) params as an argument.

    Params are an argument rather than a closure so that one compiled function
    serves every checkpoint: swapping weights does not trigger recompilation.
    """
    transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"][
        "mesh_model_ctor"
    ].keywords["transformer_kwargs"]
    if jax.default_backend() == "gpu":
        # The default splash attention kernel is TPU-only.
        transformer_kwargs["attention_type"] = "triblockdiag_mha"

    config_inference = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1],  # Drop ensemble wrap.
    )

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        predictor = fgn.construct_predictor(config_inference)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return xarray_jax.pmap(
        lambda params, rng, i, t, f: run_forward.apply(params, rng, i, t, f),
        dim="sample",
    )


def run_forecast(
    init: pd.Timestamp,
    *,
    args,
    task,
    output_spec,
    predictor_fn_pmap,
    weights,
    lead_hours: int,
    ensemble_size: int,
    output_path: str,
    base_seed: int,
) -> None:
    """Runs and saves one forecast. Everything reusable is passed in."""
    num_steps = lead_hours // 6
    members_per_model = ensemble_size // len(weights)
    num_devices = jax.local_device_count()

    logger.info(
        "WeatherNext 2 inference: init=%s, lead=%sh (%s steps), "
        "ensemble_size=%s (%s member(s) from each of %s checkpoints) -> %s",
        init,
        lead_hours,
        num_steps,
        ensemble_size,
        members_per_model,
        len(weights),
        output_path,
    )

    inputs_raw = load_era5_initial_conditions(
        args.era5_path, init, task.pressure_levels
    )

    with timed("input_extraction"):
        batch = build_batch(inputs_raw, task, num_steps)
        inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=slice("6h", f"{lead_hours}h"),
            **dataclasses.asdict(task),
        )
        inputs = inputs.compute()
        forcings = forcings.compute()
    logger.info(
        "Inputs: %s | Targets: %s | Forcings: %s",
        dict(inputs.sizes),
        dict(targets_template.sizes),
        dict(forcings.sizes),
    )

    # Fold the init date into the base seed: each date draws its own member
    # noise, reproducibly from the date alone, so no two dates in the archive
    # share a noise sequence and a single date can be reproduced in isolation.
    rng = jax.random.fold_in(
        jax.random.PRNGKey(base_seed), int(init.strftime("%Y%m%d%H"))
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    global_attrs = {
        "init_time": str(init),
        "lead_hours": lead_hours,
        "model": ", ".join(os.path.splitext(os.path.basename(w))[0] for w in weights),
        "members_per_model": members_per_model,
        "initial_conditions": args.era5_path,
        "created_by": "wn2_inference.py",
    }
    writer = None
    if args.format == "zarr":
        writer = zarr_stream.ZarrForecastWriter(
            output_path,
            output_spec,
            init_time=init,
            timestep=TIMESTEP,
            n_steps=num_steps,
            lat=inputs["lat"].values,
            lon=inputs["lon"].values,
            n_members=ensemble_size,
            global_attrs=global_attrs,
        )

    logger.info("Starting rollout ...")
    rollout_start = time_module.perf_counter()
    chunk_end = rollout_start
    first_chunk_end = None
    chunks = []
    model_descriptions = []
    total_chunks = num_steps * ensemble_size // num_devices
    chunk_count = 0
    devices = jax.local_devices()
    with writer if writer is not None else contextlib.nullcontext():
        for k, weights_path in enumerate(weights):
            with timed(f"checkpoint_load[{k}]"), open(weights_path, "rb") as f:
                ckpt = checkpoint.load(f, fgn.CheckPoint)
            model_descriptions.append(ckpt.description)
            member_offset = k * members_per_model
            logger.info(
                "Checkpoint %d/%d (%s): members %d-%d",
                k + 1,
                len(weights),
                ckpt.description,
                member_offset,
                member_offset + members_per_model - 1,
            )
            params = jax.tree_util.tree_map(
                lambda x: rollout.device_put_sharded(
                    [x] * len(devices), devices, "sample"
                ),
                ckpt.params,
            )
            predictor_fn = functools.partial(predictor_fn_pmap, params)
            # Fold in the global member index so the noise draw of member i is the
            # same regardless of how members are split across checkpoints.
            rngs = np.stack(
                [
                    jax.random.fold_in(rng, member_offset + j)
                    for j in range(members_per_model)
                ],
                axis=0,
            )

            # Reset the baseline so checkpoint load time isn't attributed to the
            # first chunk of this model.
            chunk_end = time_module.perf_counter()
            for i, chunk in enumerate(
                rollout.chunked_prediction_generator_multiple_runs(
                    predictor_fn=predictor_fn,
                    rngs=rngs,
                    inputs=inputs,
                    targets_template=targets_template,
                    forcings=forcings,
                    num_steps_per_chunk=1,
                    num_samples=members_per_model,
                    pmap_devices=jax.local_devices(),
                )
            ):
                chunk = jax.device_get(chunk)
                chunk.coords["sample"] = chunk.coords["sample"].values + member_offset
                group_start = member_offset + (i // num_steps) * num_devices
                if writer is not None:
                    if i % num_steps == 0:
                        writer.start_members(
                            range(group_start, group_start + num_devices)
                        )
                    writer.submit(
                        chunk.squeeze("batch", drop=True)
                        .squeeze("time", drop=True)
                        .rename({"sample": "number"})
                    )
                else:
                    chunks.append(chunk)
                del chunk
                previous_end, chunk_end = chunk_end, time_module.perf_counter()
                if first_chunk_end is None:
                    first_chunk_end = chunk_end
                chunk_count += 1
                perf_logger.info(
                    "chunk %d/%d (members %d-%d, lead %dh): %.2fs%s",
                    chunk_count,
                    total_chunks,
                    group_start,
                    group_start + num_devices - 1,
                    6 * (i % num_steps + 1),
                    chunk_end - previous_end,
                    " (includes JIT compilation)" if chunk_count == 1 else "",
                )
        if writer is not None:
            with timed("zarr_close"):
                writer.close()
    rollout_total = time_module.perf_counter() - rollout_start
    perf_logger.info("rollout: %.1fs total for %d steps", rollout_total, total_chunks)
    if total_chunks > 1:
        perf_logger.info(
            "rollout rate after first chunk (incl. checkpoint swaps): %.2fs/step",
            (chunk_end - first_chunk_end) / (total_chunks - 1),
        )
    resource_logger.info("post-rollout: %s", resource_snapshot())
    logger.info("Rollout finished in %.1fs", rollout_total)

    if writer is not None:
        logger.info(
            "Forecast summary:\n%s",
            xarray.open_zarr(output_path, decode_timedelta=True),
        )
        return

    with timed("combine_chunks"):
        predictions = xarray.combine_by_coords(chunks)

    # Attach valid-time info and tidy up dims for saving.
    predictions = predictions.squeeze("batch", drop=True)
    predictions = predictions.assign_coords(
        datetime=("time", init.to_numpy() + predictions["time"].values),
        model=("sample", np.repeat(model_descriptions, members_per_model)),
    )
    if not args.save_cyclone_vars:
        predictions = predictions.drop_vars(
            [v for v in predictions.data_vars if v.startswith("cyclone_")]
        )
    predictions.attrs.update(global_attrs)

    logger.info("Saving forecast to %s ...", output_path)
    with timed("netcdf_save"):
        predictions.to_netcdf(output_path)
    logger.info("Saved forecast to %s", output_path)
    logger.info("Forecast summary:\n%s", predictions)


def main():
    args = parse_args()
    setup_logging()

    cfg = reforecast_config.load(args.config)
    if args.data_dir:
        cfg = dataclasses.replace(
            cfg, data=dataclasses.replace(cfg.data, data_dir=args.data_dir)
        )
    if cfg.data.group_permissions:
        reforecast_config.apply_group_permissions()

    lead_hours = args.lead_hours or cfg.reforecast.lead_hours
    ensemble_size = args.ensemble_size or cfg.reforecast.n_members
    weights = list(args.weights or cfg.reforecast.weights)
    config_name = args.config_name or cfg.reforecast.config_name
    base_seed = cfg.reforecast.seed if args.seed is None else args.seed
    extension = "zarr" if args.format == "zarr" else "nc"

    if lead_hours % 6 != 0 or lead_hours < 6:
        raise ValueError("lead_hours must be a positive multiple of 6.")
    if ensemble_size % len(weights) != 0:
        raise ValueError(
            f"ensemble_size ({ensemble_size}) must be a multiple of the number "
            f"of weights files ({len(weights)})."
        )
    members_per_model = ensemble_size // len(weights)
    num_devices = jax.local_device_count()
    if members_per_model % num_devices != 0:
        raise ValueError(
            f"Members per checkpoint ({members_per_model}) must be a multiple of "
            f"the number of devices ({num_devices}), which run members in "
            f"parallel via pmap. Use an ensemble_size in multiples of "
            f"{len(weights) * num_devices}."
        )

    # `log_dir` defaults to the config's; an explicit empty string opts out.
    log_dir = cfg.resources.log_dir if args.log_dir is None else args.log_dir or None

    with log_files(log_dir, cfg.data.group_permissions):
        logger.info(
            "JAX backend: %s, devices: %s", jax.default_backend(), jax.devices()
        )
        inits = resolve_init_times(args, cfg)
        if args.output and len(inits) > 1:
            raise ValueError(
                f"--output takes a single init, but {len(inits)} were selected; "
                "use --data_dir to redirect a multi-date run."
            )
        planned, skipped = [], []
        for init in inits:
            path = (
                args.output
                if args.output
                else str(cfg.store_path(init, extension))
            )
            if args.skip_existing and os.path.exists(path):
                skipped.append(init)
            else:
                planned.append((init, path))
        if skipped:
            logger.info(
                "Skipping %d init(s) with a published store already: %s",
                len(skipped),
                _summarize_dates(skipped),
            )
        if not planned:
            logger.info("Nothing to do: every selected init is already published.")
            return 0
        logger.info(
            "Running %d init(s): %s", len(planned), _summarize_dates(
                [init for init, _ in planned]
            )
        )

        monitor = ResourceMonitor(args.resource_log_interval)
        monitor.start()
        run_start = time_module.perf_counter()
        failures = []
        try:
            with timed("config_load"):
                config = fiddle_config_io.get_fiddle_config_by_name(config_name)
            task = config.task

            output_spec = None
            if args.format == "zarr":
                # Re-parse now that the model's levels are known, so a config
                # asking for a level the model lacks fails before any rollout.
                output_spec = reforecast_config.load(
                    args.config, available_levels=task.pressure_levels
                ).output_spec
                logger.info(
                    "Zarr output spec (%s): %s",
                    args.config,
                    ", ".join(o.name for o in output_spec.outputs),
                )

            # Built once: the compiled predictor is reused by every init, so
            # only the first date pays the ~3 min compilation.
            predictor_fn_pmap = build_predictor_fn(config)

            for index, (init, output_path) in enumerate(planned, start=1):
                logger.info(
                    "=== init %d/%d: %s ===", index, len(planned), init
                )
                date_log_dir = (
                    pathlib.Path(log_dir) / init.strftime("%Y-%m-%dT%H")
                    if log_dir
                    else None
                )
                start = time_module.perf_counter()
                try:
                    with log_files(date_log_dir, cfg.data.group_permissions):
                        run_forecast(
                            init,
                            args=args,
                            task=task,
                            output_spec=output_spec,
                            predictor_fn_pmap=predictor_fn_pmap,
                            weights=weights,
                            lead_hours=lead_hours,
                            ensemble_size=ensemble_size,
                            output_path=output_path,
                            base_seed=base_seed,
                        )
                except Exception:  # noqa: BLE001 - one bad date must not sink the job.
                    failures.append(init)
                    logger.exception(
                        "init %s FAILED after %.1fs; continuing with the "
                        "remaining date(s)",
                        init,
                        time_module.perf_counter() - start,
                    )
                else:
                    logger.info(
                        "init %s done in %.1fs (%d/%d)",
                        init,
                        time_module.perf_counter() - start,
                        index,
                        len(planned),
                    )
        finally:
            monitor.stop()

        elapsed = time_module.perf_counter() - run_start
        logger.info(
            "Finished %d/%d init(s) in %.1fs (%.2f h/init)",
            len(planned) - len(failures),
            len(planned),
            elapsed,
            elapsed / 3600 / len(planned),
        )
        if failures:
            logger.error(
                "%d init(s) failed: %s", len(failures), _summarize_dates(failures)
            )
            return 1
    return 0


def _summarize_dates(dates) -> str:
    """Compact rendering of a date list for one log line."""
    formatted = [pd.Timestamp(d).strftime("%Y-%m-%dT%H") for d in dates]
    if len(formatted) <= 6:
        return ", ".join(formatted)
    return f"{formatted[0]} ... {formatted[-1]} ({len(formatted)} dates)"


if __name__ == "__main__":
    sys.exit(main())
