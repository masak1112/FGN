"""Streams WeatherNext 2 forecast steps into a sharded zarr v3 store.

Design contract (mirrors reforecast_config.yaml):

1.  The store is preallocated once from a lazy template with
    `to_zarr(..., compute=False)` — metadata only. All forecast data is then
    written append-only by a built-in shard streamer: encoded inner chunks
    are appended to each shard file as they complete, and the shard's index
    footer is written when its last chunk lands. The store is never read (and
    never rewritten) while it is being produced — no read-modify-write, no
    appends through zarr, no metadata rewrites. Attributes therefore all live
    on the template, the `prediction_timedelta_daily` description included.
2.  `submit()` pushes a forecast step onto a bounded queue and returns; a
    daemon worker thread reorients, aggregates, compresses and writes. Peak
    memory tracks the step and chunk sizes, not forecast length or ensemble
    size; a slow disk blocks the producer instead of growing the heap. Worker
    exceptions are stashed, the queue is drained so a blocked producer
    unblocks, and the error is re-raised on the producer thread at the next
    call.
3.  Three independently-filling dims, all chunk-aligned: `number` (member
    groups written sequentially; chunk and shard size 1 so each member owns
    its own shard files), `prediction_timedelta` and
    `prediction_timedelta_daily` (both buffered to one inner chunk before
    being encoded and appended — a chunk is the atomic unit of encoding, so
    nothing smaller can be written without re-reading).
4.  Daily aggregates fold step-by-step through `_Reducer`s
    (identity/combine/finalize); raw steps are never held. Accumulation is
    float64, cast to float32 only at finalize. `np.minimum`/`np.maximum`
    propagate NaN deliberately (skipna=False semantics). Steps on incomplete
    edge days are dropped.
5.  Everything goes to `<name>_partial.zarr`; only a clean `close()` — after
    completeness checks — renames it to the final path. On error the staged
    store is left in place with a warning rather than published.

The shard streamer writes the zarr v3 `sharding_indexed` binary format
directly, deriving shapes, codecs and keys from the template's own array
metadata so the two cannot drift apart. Correctness is proven by reading
stores back through zarr itself in tests/test_zarr_stream.py.
"""

import dataclasses
import itertools
import json
import logging
import math
import os
import pathlib
import queue
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from typing import Self

import crc32c
import dask.array
import numcodecs
import numpy as np
import pandas as pd
import xarray
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import BitRound

import zarr_config

logger = logging.getLogger("wn2.zarr")
perf_logger = logging.getLogger("wn2.perf")

_NATIVE_DIM = "prediction_timedelta"
_DAILY_DIM = "prediction_timedelta_daily"
_DAILY_DESCRIPTION = (
    "Start of the UTC calendar day (as an offset from forecast init time) "
    "over which daily aggregations are computed. A step valid at t covers "
    "(t-6h, t], so a day owns the four steps valid at 06/12/18 UTC of that "
    "date and 00 UTC of the next. Days without all their steps in the "
    "forecast are dropped."
)

_MISSING_CHUNK = 2**64 - 1  # Sharding-spec marker for an absent inner chunk.


@dataclasses.dataclass(frozen=True)
class _Reducer:
    """An aggregation expressed as an incremental fold."""

    identity: Callable[[np.ndarray], np.ndarray]
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray]
    finalize: Callable[[np.ndarray, int], np.ndarray]


def _cast(acc: np.ndarray, _: int) -> np.ndarray:
    return acc.astype(np.float32)


# Accumulators are float64 and cast to float32 only at finalize, making this
# path slightly *more* accurate than a float32 batch computation. min/max use
# the NaN-propagating numpy ufuncs: a day with a missing sample has no valid
# extreme (skipna=False semantics).
_REDUCERS = {
    zarr_config.DAILY_MEAN: _Reducer(
        identity=lambda x: x.astype(np.float64),
        combine=np.add,
        finalize=lambda acc, n: (acc / n).astype(np.float32),
    ),
    zarr_config.DAILY_SUM: _Reducer(
        identity=lambda x: x.astype(np.float64), combine=np.add, finalize=_cast
    ),
    zarr_config.DAILY_MIN: _Reducer(
        identity=lambda x: x.astype(np.float64), combine=np.minimum, finalize=_cast
    ),
    zarr_config.DAILY_MAX: _Reducer(
        identity=lambda x: x.astype(np.float64), combine=np.maximum, finalize=_cast
    ),
}


def _reducer(aggregation: str) -> _Reducer:
    try:
        return _REDUCERS[aggregation]
    except KeyError:
        raise KeyError(
            f"No reducer registered for aggregation '{aggregation}'. "
            f"Known: {sorted(_REDUCERS)}."
        ) from None


class _ChunkEncoder:
    """Encodes one inner chunk per the array's codec metadata.

    Supports the chain this writer's own template produces:
    [numcodecs.bitround?] -> bytes(little) -> [blosc?].
    """

    def __init__(self, codecs_meta: list, dtype: str):
        chain = list(codecs_meta)
        self._bitround = None
        if chain and chain[0]["name"] == "numcodecs.bitround":
            self._bitround = numcodecs.BitRound(**chain.pop(0).get("configuration", {}))
        if (
            not chain
            or chain[0]["name"] != "bytes"
            or chain[0].get("configuration", {}).get("endian") != "little"
        ):
            raise ValueError(
                f"Unsupported codec chain {codecs_meta}: expected "
                "little-endian 'bytes' after optional bitround."
            )
        chain.pop(0)
        self._blosc = None
        if chain:
            blosc_meta = chain.pop(0)
            if blosc_meta["name"] != "blosc" or chain:
                raise ValueError(
                    f"Unsupported codec chain {codecs_meta}: only a "
                    "single trailing blosc compressor is supported."
                )
            conf = blosc_meta["configuration"]
            self._blosc = numcodecs.Blosc(
                cname=conf["cname"],
                clevel=conf["clevel"],
                shuffle={
                    "noshuffle": numcodecs.Blosc.NOSHUFFLE,
                    "shuffle": numcodecs.Blosc.SHUFFLE,
                    "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
                }[conf["shuffle"]],
                blocksize=conf.get("blocksize", 0),
            )
        self._dtype = np.dtype(dtype).newbyteorder("<")

    def encode(self, chunk: np.ndarray) -> bytes:
        chunk = np.ascontiguousarray(chunk, dtype=self._dtype)
        if self._bitround is not None:
            # BitRound mutates in place and returns an integer *view* of the
            # rounded float bits; copy first, reinterpret after.
            rounded = np.asarray(self._bitround.encode(chunk.copy()))
            chunk = rounded.view(self._dtype)
        if self._blosc is None:
            return chunk.tobytes()
        return self._blosc.encode(chunk)


class _ShardWriter:
    """Appends encoded inner chunks to one shard file, index footer last.

    Implements the zarr v3 `sharding_indexed` layout with the default
    bytes(little)+crc32c index codecs and index_location='end': encoded chunks
    back to back, then a (chunks_per_shard, 2) uint64 (offset, nbytes) index
    followed by its crc32c checksum.
    """

    def __init__(self, path: pathlib.Path, chunks_per_shard: tuple[int, ...]):
        path.parent.mkdir(parents=True, exist_ok=True)
        # The handle outlives __init__ by design: this class *is* the shard's
        # resource manager, closed by finalize() or discard().
        self._file = open(path, "wb")  # noqa: SIM115
        self._index = np.full(chunks_per_shard + (2,), _MISSING_CHUNK, dtype="<u8")
        self._offset = 0

    def append(self, chunk_in_shard: tuple[int, ...], data: bytes) -> None:
        self._file.write(data)
        self._index[chunk_in_shard] = (self._offset, len(data))
        self._offset += len(data)

    def finalize(self) -> None:
        index_bytes = self._index.tobytes(order="C")
        self._file.write(index_bytes)
        self._file.write(crc32c.crc32c(index_bytes).to_bytes(4, "little"))
        self._file.close()

    def discard(self) -> None:
        self._file.close()


class _ArrayStreamer:
    """Streams chunk-aligned slabs of one array into its shard files.

    All layout information (shard/chunk grids, codecs, keys, fill value) is
    parsed from the array's zarr.json, written moments earlier by the template,
    so streamer and metadata cannot disagree.
    """

    def __init__(self, store_path: pathlib.Path, name: str):
        self._array_path = store_path / name
        with open(self._array_path / "zarr.json") as f:
            meta = json.load(f)
        if meta["zarr_format"] != 3 or meta["node_type"] != "array":
            raise ValueError(f"{name}: not a zarr v3 array.")
        key_encoding = meta["chunk_key_encoding"]
        if (
            key_encoding["name"] != "default"
            or key_encoding.get("configuration", {}).get("separator") != "/"
        ):
            raise ValueError(f"{name}: unsupported chunk key encoding {key_encoding}.")
        (sharding,) = meta["codecs"]  # Exactly one top-level codec expected.
        if sharding["name"] != "sharding_indexed":
            raise ValueError(
                f"{name}: expected a sharding_indexed codec, found {sharding['name']}."
            )
        config = sharding["configuration"]
        index_names = [c["name"] for c in config["index_codecs"]]
        if index_names != ["bytes", "crc32c"] or (config["index_location"] != "end"):
            raise ValueError(
                f"{name}: unsupported shard index layout "
                f"({index_names}, {config['index_location']})."
            )

        self._shape = tuple(meta["shape"])
        self._shard_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        self._chunk_shape = tuple(config["chunk_shape"])
        self._chunks_per_shard = tuple(
            s // c for s, c in zip(self._shard_shape, self._chunk_shape)
        )
        self._n_chunks = tuple(
            math.ceil(d / c) for d, c in zip(self._shape, self._chunk_shape)
        )
        self._fill = float(meta["fill_value"])
        self._encoder = _ChunkEncoder(config["codecs"], meta["data_type"])
        self._writers: dict[tuple[int, ...], _ShardWriter] = {}
        self._remaining: dict[tuple[int, ...], int] = {}

    def _valid_chunks_in_shard(self, shard: tuple[int, ...]) -> int:
        """How many inner chunks of this shard intersect the array bounds."""
        count = 1
        for s, per_shard, n_valid in zip(shard, self._chunks_per_shard, self._n_chunks):
            count *= max(0, min((s + 1) * per_shard, n_valid) - s * per_shard)
        return count

    def _writer_for(self, shard: tuple[int, ...]) -> _ShardWriter:
        if shard not in self._writers:
            key = pathlib.Path("c", *map(str, shard))
            self._writers[shard] = _ShardWriter(
                self._array_path / key, self._chunks_per_shard
            )
            self._remaining[shard] = self._valid_chunks_in_shard(shard)
        return self._writers[shard]

    def write_slab(self, origin: tuple[int, ...], slab: np.ndarray) -> None:
        """Encodes and appends every inner chunk covered by the slab.

        The slab must be chunk-aligned: on each dim, `origin` is a multiple of
        the chunk size, and the slab either ends on a chunk boundary or at the
        array edge (edge chunks are padded with the fill value).
        """
        ends = tuple(o + s for o, s in zip(origin, slab.shape))
        for dim, (o, end, chunk, size) in enumerate(
            zip(origin, ends, self._chunk_shape, self._shape)
        ):
            if o % chunk != 0 or end > size or (end % chunk != 0 and end != size):
                raise ValueError(
                    f"Slab [{o}:{end}] on dim {dim} is not chunk-aligned "
                    f"(chunk {chunk}, dim size {size})."
                )

        chunk_ranges = [
            range(o // c, math.ceil(e / c))
            for o, e, c in zip(origin, ends, self._chunk_shape)
        ]
        for chunk_coords in itertools.product(*chunk_ranges):
            part = slab[
                tuple(
                    slice(c * size - o, min((c + 1) * size, e) - o)
                    for c, size, o, e in zip(
                        chunk_coords, self._chunk_shape, origin, ends
                    )
                )
            ]
            if part.shape != self._chunk_shape:
                padded = np.full(self._chunk_shape, self._fill, dtype=part.dtype)
                padded[tuple(slice(0, n) for n in part.shape)] = part
                part = padded
            shard = tuple(c // p for c, p in zip(chunk_coords, self._chunks_per_shard))
            writer = self._writer_for(shard)
            writer.append(
                tuple(c % p for c, p in zip(chunk_coords, self._chunks_per_shard)),
                self._encoder.encode(part),
            )
            self._remaining[shard] -= 1
            if self._remaining[shard] == 0:
                writer.finalize()
                del self._writers[shard], self._remaining[shard]

    @property
    def open_shards(self) -> int:
        return len(self._writers)

    def discard_open(self) -> None:
        for writer in self._writers.values():
            writer.discard()
        self._writers.clear()
        self._remaining.clear()


_STOP_FLUSH = "stop_flush"  # Clean shutdown: flush the last group first.
_STOP_NOW = "stop_now"  # Abort: exit without flushing.


class ZarrForecastWriter:
    """Streams member-group forecast steps into a preallocated zarr store.

    Usage:
      with ZarrForecastWriter(...) as writer:
        for group of members:
          writer.start_members(members)
          for each forecast step:            # in lead-time order
            writer.submit(step_dataset)      # dims: (number, [level,] lat, lon)

    Member groups must be submitted sequentially and contiguously from member 0.
    For deterministic runs pass n_members=None: the store has no `number` dim
    and `start_members` must not be called.
    """

    def __init__(
        self,
        save_path: str,
        spec: zarr_config.OutputSpec,
        *,
        init_time: pd.Timestamp,
        timestep: pd.Timedelta,
        n_steps: int,
        lat: np.ndarray,
        lon: np.ndarray,
        n_members: int | None,
        global_attrs: dict | None = None,
    ):
        self._save_path = pathlib.Path(save_path)
        if self._save_path.exists():
            raise FileExistsError(
                f"{save_path} already exists; refusing to overwrite a published store."
            )
        self._staging_path = self._save_path.with_name(
            self._save_path.stem + "_partial" + self._save_path.suffix
        )
        self._spec = spec
        self._init_time = pd.Timestamp(init_time)
        self._timestep = pd.Timedelta(timestep)
        self._n_steps = int(n_steps)
        self._n_members = n_members

        # The store follows the reference archive's grid convention: lat strictly
        # descending (90 -> -90). Incoming steps in either orientation are
        # reorientated by the worker.
        lat = np.asarray(lat, dtype=np.float32)
        if lat[0] < lat[-1]:
            lat = lat[::-1].copy()
        self._lat = lat
        self._lon = np.asarray(lon, dtype=np.float32)

        self._global_attrs = dict(global_attrs or {})
        self._leads = pd.TimedeltaIndex(
            self._timestep * np.arange(1, self._n_steps + 1)
        )
        self._day_of_step, self._daily_coord, self._steps_per_day = (
            self._daily_windows()
        )

        # Producer-side bookkeeping for the sequential-group contract.
        self._next_member = 0  # Next member index expected to start.
        self._group: tuple[int, ...] | None = None
        self._submitted = 0  # Steps submitted for the current group.
        self._closed = False

        self._error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=8)

        self._prepare_store()
        self._streamers = {
            output.name: _ArrayStreamer(self._staging_path, output.name)
            for output in self._spec.outputs
        }
        self._worker = threading.Thread(
            target=self._run_worker, name="zarr-stream-worker", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Producer API                                                        #
    # ------------------------------------------------------------------ #

    def start_members(self, members: Sequence[int]) -> None:
        """Begins a new contiguous group of ensemble members."""
        self._check_error()
        if self._n_members is None:
            raise ValueError(
                "start_members is not applicable: this writer was "
                "created with n_members=None (deterministic mode)."
            )
        members = tuple(int(m) for m in members)
        if not members:
            raise ValueError("Empty member group.")
        if list(members) != list(range(members[0], members[-1] + 1)):
            raise ValueError(f"Member group {members} is not contiguous ascending.")
        if members[0] != self._next_member:
            raise ValueError(
                f"Member group {members} out of order: expected the next group to "
                f"start at member {self._next_member} (repeats and gaps are "
                "rejected)."
            )
        if members[-1] >= self._n_members:
            raise ValueError(
                f"Member group {members} out of range for n_members={self._n_members}."
            )
        if self._group is not None and self._submitted != self._n_steps:
            raise ValueError(
                f"Members {self._group} are short: only {self._submitted} of "
                f"{self._n_steps} steps were submitted."
            )
        self._group = members
        self._next_member = members[-1] + 1
        self._submitted = 0
        self._put(("start", members))

    def submit(self, data: xarray.Dataset) -> None:
        """Enqueues one forecast step (the next lead time) for the active group.

        `data` holds the model variables named in the spec, with dims
        (number, [level,] lat, lon) — or ([level,] lat, lon) when n_members is
        None — matching the grid the writer was constructed with, in either lat
        orientation.
        """
        self._check_error()
        if self._n_members is None:
            if self._group is None:
                self._group = ()
                self._put(("start", ()))
        elif self._group is None:
            raise ValueError("submit() before start_members().")
        if self._submitted >= self._n_steps:
            raise ValueError(
                f"Members {self._group} already submitted all {self._n_steps} steps."
            )
        step_index = self._submitted
        self._submitted += 1
        # Only the model variables the spec needs travel on the queue.
        self._put(("step", (step_index, data[list(self._spec.model_variables)])))

    def close(self) -> None:
        """Flushes, verifies completeness and publishes the store."""
        if self._closed:
            return
        self._check_error()
        if self._n_members is not None and self._next_member != self._n_members:
            raise ValueError(
                f"Cannot publish: members {self._next_member}.."
                f"{self._n_members - 1} never started."
            )
        if self._submitted != self._n_steps:
            raise ValueError(
                f"Cannot publish: the last member group submitted only "
                f"{self._submitted} of {self._n_steps} steps."
            )
        self._put((_STOP_FLUSH, None))
        self._worker.join()
        self._check_error()
        self._closed = True
        os.rename(self._staging_path, self._save_path)
        logger.info("Published %s", self._save_path)

    def abort(self) -> None:
        """Stops the worker and leaves the staged store unpublished."""
        if self._closed:
            return
        self._closed = True
        # Queued steps ahead of the stop marker are still processed (or drained
        # by a failed worker); the worker exits without flushing when it reaches
        # the marker.
        self._queue.put((_STOP_NOW, None))
        self._worker.join()
        logger.warning(
            "Zarr stream aborted; incomplete staged store left at %s "
            "(not published to %s).",
            self._staging_path,
            self._save_path,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    # ------------------------------------------------------------------ #
    # Producer internals                                                  #
    # ------------------------------------------------------------------ #

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "The zarr stream worker failed; the staged store at "
                f"{self._staging_path} is incomplete."
            ) from self._error

    def _put(self, item) -> None:
        self._queue.put(item)
        # A worker that died mid-queue drains it, so this put cannot deadlock;
        # surface its error as soon as possible.
        self._check_error()

    def _daily_windows(self):
        """Maps native steps onto complete UTC calendar days.

        Returns (day_of_step, daily_coord, steps_per_day) where day_of_step[k] is
        the daily index owning native step k, or -1 for steps on incomplete edge
        days that are dropped.
        """
        day_length = pd.Timedelta("1D")
        if day_length % self._timestep != pd.Timedelta(0):
            raise ValueError(f"Timestep {self._timestep} does not divide a day.")
        steps_per_day = day_length // self._timestep

        valid = self._init_time + self._leads
        # A step valid at t covers (t-6h, t], so a step valid at 00 UTC belongs
        # to the previous day.
        owner = (valid - pd.Timedelta(1, "ns")).floor("D")
        counts = pd.Series(owner).value_counts()
        complete_days = sorted(day for day, n in counts.items() if n == steps_per_day)
        day_index = {day: i for i, day in enumerate(complete_days)}
        day_of_step = np.array([day_index.get(day, -1) for day in owner])
        daily_coord = (pd.DatetimeIndex(complete_days) - self._init_time).to_numpy()
        return day_of_step, daily_coord, int(steps_per_day)

    def _dims_of(self, output: zarr_config.OutputVariable) -> tuple[str, ...]:
        time_dim = _NATIVE_DIM if output.is_native else _DAILY_DIM
        if self._n_members is None:
            return ("time", time_dim, "lat", "lon")
        return ("time", "number", time_dim, "lat", "lon")

    def _prepare_store(self) -> None:
        """Preallocates the store: full metadata and coords, no data.

        The streamed writes never touch attributes, so everything — variable
        attrs and the daily-coordinate description included — must be set here.
        """
        if self._staging_path.exists():
            logger.warning("Removing stale staged store %s", self._staging_path)
            shutil.rmtree(self._staging_path)

        sizes = {
            "time": 1,
            "number": self._n_members,
            _NATIVE_DIM: self._n_steps,
            _DAILY_DIM: len(self._daily_coord),
            "lat": len(self._lat),
            "lon": len(self._lon),
        }
        coords = {
            "time": np.array([self._init_time.to_datetime64()]),
            _NATIVE_DIM: self._leads.to_numpy(),
            _DAILY_DIM: self._daily_coord,
            "lat": self._lat,
            "lon": self._lon,
        }
        if self._n_members is not None:
            coords["number"] = np.arange(self._n_members)

        template_vars, encoding = {}, {}
        for output in self._spec.outputs:
            dims = self._dims_of(output)
            shape = tuple(sizes[d] for d in dims)
            template_vars[output.name] = xarray.DataArray(
                dask.array.zeros(shape, chunks=-1, dtype=np.float32),
                dims=dims,
                attrs={
                    "units": output.units,
                    "aggregation": output.aggregation,
                    "source": output.model_var
                    + (f" at {output.level} hPa" if output.level else ""),
                },
            )
            spec_encoding = self._spec.encoding
            var_encoding = {
                "chunks": tuple(spec_encoding.chunks[d] for d in dims),
                "shards": tuple(spec_encoding.shards[d] for d in dims),
                "compressors": (BloscCodec(**spec_encoding.compressor),),
            }
            keepbits = spec_encoding.keepbits_for(output.name)
            if keepbits is not None:
                var_encoding["filters"] = (BitRound(keepbits=int(keepbits)),)
            encoding[output.name] = var_encoding

        template = xarray.Dataset(template_vars, coords=coords)
        template[_DAILY_DIM].attrs["description"] = _DAILY_DESCRIPTION
        template.attrs.update(self._global_attrs)

        start = time.perf_counter()
        template.to_zarr(
            self._staging_path,
            mode="w-",
            zarr_format=3,
            encoding=encoding,
            compute=False,
        )
        perf_logger.info(
            "zarr template preallocation: %.2fs", time.perf_counter() - start
        )

    # ------------------------------------------------------------------ #
    # Worker                                                              #
    # ------------------------------------------------------------------ #

    def _run_worker(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get()
                if kind == _STOP_NOW:
                    self._discard_open_shards()
                    return
                if kind == _STOP_FLUSH:
                    self._flush_group()
                    return
                if kind == "start":
                    self._flush_group()
                    self._begin_group(payload)
                else:
                    self._process_step(*payload)
        except BaseException as e:  # pylint: disable=broad-except
            self._error = e
            logger.exception("Zarr stream worker failed")
            self._discard_open_shards()
            # Unblock a producer waiting on the bounded queue.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def _discard_open_shards(self) -> None:
        for streamer in self._streamers.values():
            streamer.discard_open()

    def _begin_group(self, members: tuple[int, ...]) -> None:
        self._worker_group = members
        self._native_buffer = {o.name: [] for o in self._spec.native_outputs}
        self._native_flushed = 0
        self._daily_buffer = {o.name: [] for o in self._spec.daily_outputs}
        self._daily_flushed = 0
        self._daily_acc: dict[int, dict[str, np.ndarray]] = {}
        self._daily_count: dict[int, int] = {}
        self._flip_lat: bool | None = None

    def _extract(
        self, data: xarray.Dataset, output: zarr_config.OutputVariable
    ) -> np.ndarray:
        """Pulls one output's field from a step as (n_group?, lat, lon) float32."""
        da = data[output.model_var]
        if output.level is not None:
            da = da.sel(level=output.level)
        expected_dims = (
            ("lat", "lon") if self._n_members is None else ("number", "lat", "lon")
        )
        da = da.transpose(*expected_dims)

        if self._flip_lat is None:
            data_lat = np.asarray(da["lat"].values, dtype=np.float32)
            if np.array_equal(data_lat, self._lat):
                self._flip_lat = False
            elif np.array_equal(data_lat[::-1], self._lat):
                self._flip_lat = True
            else:
                raise ValueError(
                    "Submitted latitudes match the store grid in neither orientation."
                )
            if not np.array_equal(
                np.asarray(da["lon"].values, dtype=np.float32), self._lon
            ):
                raise ValueError("Submitted longitudes do not match the store grid.")

        values = da.values
        if self._flip_lat:
            values = values[..., ::-1, :]
        # Copy: level selection and the lat flip produce views that would pin the
        # step's full arrays in memory for as long as they sit in a buffer.
        return np.ascontiguousarray(values, dtype=np.float32)

    def _process_step(self, step_index: int, data: xarray.Dataset) -> None:
        for output in self._spec.native_outputs:
            self._native_buffer[output.name].append(self._extract(data, output))
        buffered = len(next(iter(self._native_buffer.values()), []))
        if buffered and buffered == self._spec.encoding.chunks[_NATIVE_DIM]:
            self._flush(self._native_buffer, "_native_flushed", "steps")

        day = self._day_of_step[step_index]
        if day < 0 or not self._spec.daily_outputs:
            return  # Incomplete edge day: dropped.
        if day not in self._daily_acc:
            self._daily_acc[day] = {}
            self._daily_count[day] = 0
        acc = self._daily_acc[day]
        for output in self._spec.daily_outputs:
            reducer = _reducer(output.aggregation)
            step = self._extract(data, output)
            if output.name not in acc:
                acc[output.name] = reducer.identity(step)
            else:
                acc[output.name] = reducer.combine(acc[output.name], step)
        self._daily_count[day] += 1
        if self._daily_count[day] == self._steps_per_day:
            self._finalize_day(day)

    def _finalize_day(self, day: int) -> None:
        """Moves a completed day from its accumulators into the write buffer."""
        acc = self._daily_acc.pop(day)
        self._daily_count.pop(day)
        expected = self._daily_flushed + len(next(iter(self._daily_buffer.values())))
        if day != expected:
            raise RuntimeError(
                f"Day {day} completed out of order (expected {expected})."
            )
        for output in self._spec.daily_outputs:
            self._daily_buffer[output.name].append(
                _reducer(output.aggregation).finalize(
                    acc[output.name], self._steps_per_day
                )
            )
        del acc
        buffered = len(next(iter(self._daily_buffer.values())))
        if buffered == self._spec.encoding.chunks[_DAILY_DIM]:
            self._flush(self._daily_buffer, "_daily_flushed", "days")

    def _flush(
        self, buffer: dict[str, list[np.ndarray]], flushed_attr: str, label: str
    ) -> None:
        """Encodes and appends one buffered chunk-slab per output variable."""
        buffered = len(next(iter(buffer.values()), []))
        if not buffered:
            return
        start_index = getattr(self, flushed_attr)
        setattr(self, flushed_attr, start_index + buffered)
        start = time.perf_counter()
        for name, steps in buffer.items():
            slab = np.stack(steps, axis=1 if self._n_members is not None else 0)
            slab = slab[np.newaxis]  # Leading time dim of size 1.
            steps.clear()
            if self._n_members is not None:
                origin = (0, self._worker_group[0], start_index, 0, 0)
            else:
                origin = (0, start_index, 0, 0)
            self._streamers[name].write_slab(origin, slab)
        members = (
            f" members {self._worker_group[0]}-{self._worker_group[-1]},"
            if self._worker_group
            else ""
        )
        perf_logger.info(
            "zarr append:%s %s %d-%d: %.2fs",
            members,
            label,
            start_index,
            getattr(self, flushed_attr) - 1,
            time.perf_counter() - start,
        )

    def _flush_group(self) -> None:
        if getattr(self, "_worker_group", None) is None:
            return
        self._flush(self._native_buffer, "_native_flushed", "steps")
        self._flush(self._daily_buffer, "_daily_flushed", "days")
        if self._daily_acc:
            # Accumulators for days that never completed: edge days, dropped.
            logger.info(
                "Dropping %d incomplete edge day(s) for members %s",
                len(self._daily_acc),
                self._worker_group,
            )
            self._daily_acc.clear()
            self._daily_count.clear()
        open_shards = sum(s.open_shards for s in self._streamers.values())
        if open_shards:
            raise RuntimeError(
                f"{open_shards} shard file(s) still open after a complete member "
                "group; shard completion bookkeeping is inconsistent."
            )
        self._worker_group = None
