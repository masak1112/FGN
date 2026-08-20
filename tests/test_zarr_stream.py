"""Tests for zarr_config parsing and the ZarrForecastWriter.

The writer tests compare against a "batch oracle": the same aggregations
computed in one shot with plain numpy/xarray (float32, skipna=False).
"""

import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray

import reforecast_config
import zarr_config
import zarr_stream

INIT = pd.Timestamp("2026-06-25T00")
STEP = pd.Timedelta("6h")
LAT_DESC = np.linspace(90, -90, 7, dtype=np.float32)
LON = np.arange(0, 360, 60, dtype=np.float32)
LEVELS = [500, 850]


def write_config(tmp_path, body) -> str:
  path = tmp_path / "config.yaml"
  path.write_text(textwrap.dedent(body))
  return str(path)


SMALL_CONFIG = """
variables:
  2t:    {units: K,          aggregations: [native, daily_min, daily_max]}
  msl:   {units: Pa,         aggregations: [native]}
  t_850: {units: K,          aggregations: [daily_mean]}
  z_500: {units: m**2 s**-2, aggregations: [daily_mean]}
  tp:    {units: m,          aggregations: [daily_sum]}
encoding:
  chunks: {number: 1, prediction_timedelta: 5, prediction_timedelta_daily: 2,
           lat: 3, lon: 6}
  shards: {number: 1, prediction_timedelta: 10, prediction_timedelta_daily: 2,
           lat: 6, lon: 6}
"""


# --------------------------------------------------------------------- #
# Config parsing                                                         #
# --------------------------------------------------------------------- #


class TestConfig:

  def test_example_config_parses(self):
    # The project's one config; its extra sections belong to reforecast_config,
    # which hands the variables/encoding pair back through zarr_config.parse.
    spec = reforecast_config.load("reforecast_config.yaml",
                                  available_levels=LEVELS + [
        50, 100, 150, 200, 250, 300, 400, 600, 700, 925, 1000]).output_spec
    names = [o.name for o in spec.outputs]
    # Single aggregation -> bare name, even for daily aggregations.
    assert "2t" in names and "u_50" in names and "tp" in names
    tp = next(o for o in spec.outputs if o.name == "tp")
    assert tp.aggregation == zarr_config.DAILY_SUM
    assert tp.model_var == "total_precipitation_6hr"
    assert spec.encoding.chunks["prediction_timedelta"] == 24
    assert spec.encoding.shards["lat"] == 720
    assert spec.encoding.keepbits is None

  def test_multi_aggregation_naming(self, tmp_path):
    spec = zarr_config.load(write_config(tmp_path, SMALL_CONFIG),
                            available_levels=LEVELS)
    names = {o.name for o in spec.outputs}
    # native never takes a suffix; the daily extremes do.
    assert {"2t", "2t_min", "2t_max", "msl", "t_850", "z_500", "tp"} == names
    t850 = next(o for o in spec.outputs if o.name == "t_850")
    assert t850.model_var == "temperature" and t850.level == 850

  def test_unknown_aggregation_raises(self, tmp_path):
    path = write_config(tmp_path, """
        variables:
          2t: {units: K, aggregations: [daily_median]}
        """)
    with pytest.raises(ValueError, match="unknown aggregation"):
      zarr_config.load(path)

  def test_unknown_variable_raises(self, tmp_path):
    path = write_config(tmp_path, """
        variables:
          notavar: {units: K, aggregations: [native]}
        """)
    with pytest.raises(ValueError, match="Unknown variable"):
      zarr_config.load(path)

  def test_unavailable_level_raises(self, tmp_path):
    path = write_config(tmp_path, """
        variables:
          t_875: {units: K, aggregations: [native]}
        """)
    with pytest.raises(ValueError, match="level 875"):
      zarr_config.load(path, available_levels=LEVELS)

  def test_misaligned_shards_raise(self, tmp_path):
    path = write_config(tmp_path, """
        variables:
          2t: {units: K, aggregations: [native]}
        encoding:
          chunks: {lat: 90}
          shards: {lat: 100}
        """)
    with pytest.raises(ValueError, match="multiple of the chunk"):
      zarr_config.load(path)

  def test_reducer_table_rejects_unknown(self):
    with pytest.raises(KeyError, match="No reducer"):
      zarr_stream._reducer("native")


# --------------------------------------------------------------------- #
# Writer                                                                 #
# --------------------------------------------------------------------- #


def make_step(rng, n_members=None, lat=None, nan_at=None):
  """One synthetic forecast step with the WN2 variables the config needs."""
  lat = LAT_DESC if lat is None else lat
  member_dims = () if n_members is None else ("number",)
  member_shape = () if n_members is None else (n_members,)
  shape2d = member_shape + (len(lat), len(LON))
  shape3d = member_shape + (len(LEVELS), len(lat), len(LON))

  def surface():
    return xarray.DataArray(
        rng.normal(280, 10, shape2d).astype(np.float32),
        dims=member_dims + ("lat", "lon"))

  def upper():
    return xarray.DataArray(
        rng.normal(0, 5, shape3d).astype(np.float32),
        dims=member_dims + ("level", "lat", "lon"))

  ds = xarray.Dataset(
      {"2m_temperature": surface(),
       "mean_sea_level_pressure": surface(),
       "total_precipitation_6hr": surface(),
       "temperature": upper(),
       "geopotential": upper(),
       # A variable the spec does not use; must be ignored.
       "sea_surface_temperature": surface()},
      coords={"lat": lat, "lon": LON, "level": LEVELS})
  if nan_at is not None:
    ds["2m_temperature"][nan_at] = np.nan
    level_850 = LEVELS.index(850)
    ds["temperature"][nan_at[:-2] + (level_850,) + nan_at[-2:]] = np.nan
  return ds


def run_writer(tmp_path, spec, n_steps, groups, steps, n_members):
  """Feeds `steps[member_group_index][step_index]` through a writer."""
  path = tmp_path / "out.zarr"
  with zarr_stream.ZarrForecastWriter(
      str(path), spec, init_time=INIT, timestep=STEP, n_steps=n_steps,
      lat=LAT_DESC, lon=LON, n_members=n_members,
      global_attrs={"experiment": "unit-test"}) as writer:
    for group, group_steps in zip(groups, steps):
      if n_members is not None:
        writer.start_members(group)
      for step in group_steps:
        writer.submit(step)
  return xarray.open_zarr(path, decode_timedelta=True)


def oracle_daily(steps_per_member, outputs, day_of_step):
  """Batch computation of the daily aggregations, float32, skipna=False."""
  results = {}
  for output in outputs:
    fields = []
    for step in steps_per_member:
      da = step[output.model_var]
      if output.level is not None:
        da = da.sel(level=output.level)
      fields.append(da.values)
    fields = np.stack(fields)  # (steps, lat, lon)
    days = []
    for day in range(day_of_step.max() + 1):
      window = fields[day_of_step == day]
      op = {"daily_mean": np.mean, "daily_min": np.min,
            "daily_max": np.max, "daily_sum": np.sum}[output.aggregation]
      days.append(op(window, axis=0))
    results[output.name] = np.stack(days)
  return results


class TestWriter:

  @pytest.fixture()
  def spec(self, tmp_path):
    return zarr_config.load(write_config(tmp_path, SMALL_CONFIG),
                            available_levels=LEVELS)

  def test_roundtrip_matches_oracle(self, tmp_path, spec):
    # 12 steps = 3 complete days; groups of 2 members; NaN injected into one
    # member's day-1 to verify propagation.
    rng = np.random.default_rng(0)
    n_steps, n_members = 12, 4
    groups = [(0, 1), (2, 3)]
    steps = [[make_step(rng, 2, nan_at=(1, 3, 4) if k == 5 and g == 1 else
                        None)
              for k in range(n_steps)] for g in range(2)]
    ds = run_writer(tmp_path, spec, n_steps, groups, steps, n_members)

    assert ds.sizes == {"time": 1, "number": 4, "prediction_timedelta": 12,
                        "prediction_timedelta_daily": 3, "lat": 7, "lon": 6}
    np.testing.assert_array_equal(ds.lat, LAT_DESC)
    assert list(ds.prediction_timedelta.values[:2]) == [
        np.timedelta64(6, "h"), np.timedelta64(12, "h")]
    assert list(ds.prediction_timedelta_daily.values) == [
        np.timedelta64(0, "D"), np.timedelta64(1, "D"),
        np.timedelta64(2, "D")]

    # Native fields: bit-exact float32 passthrough.
    for group_index, group in enumerate(groups):
      for member_offset, member in enumerate(group):
        got = ds["2t"].isel(time=0, number=member).values
        want = np.stack([steps[group_index][k]["2m_temperature"]
                         .values[member_offset] for k in range(n_steps)])
        np.testing.assert_array_equal(got, want)

    # Daily fields: within ~1 ulp of the float32 batch oracle (the writer's
    # float64 accumulator is the more accurate path).
    day_of_step = np.repeat(np.arange(3), 4)
    for group_index, group in enumerate(groups):
      for member_offset, member in enumerate(group):
        member_steps = [s.isel(number=member_offset)
                        for s in steps[group_index]]
        want = oracle_daily(member_steps, spec.daily_outputs, day_of_step)
        for name, expected in want.items():
          got = ds[name].isel(time=0, number=member).values
          # ~1 float32 ulp at the field's magnitude scale.
          scale = np.nanmax(np.abs(expected))
          np.testing.assert_allclose(got, expected, rtol=1.5e-7,
                                     atol=1.5e-7 * scale,
                                     equal_nan=True, err_msg=name)

    # The injected NaN (member 3, step 5 -> day 1) propagated to every daily
    # aggregation at that cell, and nowhere else.
    for name in ["2t_min", "2t_max", "t_850"]:
      field = ds[name].isel(time=0, number=3, prediction_timedelta_daily=1)
      assert np.isnan(field.values[3, 4]), name
      assert np.isfinite(np.delete(field.values.ravel(), 3 * 6 + 4)).all()

    # Attributes all came from the template.
    assert ds.attrs["experiment"] == "unit-test"
    assert ds["2t_min"].attrs["aggregation"] == "daily_min"
    assert ds["t_850"].attrs["units"] == "K"
    assert ds["t_850"].attrs["source"] == "temperature at 850 hPa"
    assert "description" in ds["prediction_timedelta_daily"].attrs
    assert ds["tp"].attrs["aggregation"] == "daily_sum"

  def test_ascending_lat_is_flipped(self, tmp_path, spec):
    rng = np.random.default_rng(1)
    lat_asc = LAT_DESC[::-1].copy()
    steps = [[make_step(rng, 1, lat=lat_asc) for _ in range(4)]]
    ds = run_writer(tmp_path, spec, 4, [(0,)], steps, 1)
    np.testing.assert_array_equal(ds.lat, LAT_DESC)
    got = ds["2t"].isel(time=0, number=0, prediction_timedelta=0).values
    want = steps[0][0]["2m_temperature"].values[0, ::-1, :]
    np.testing.assert_array_equal(got, want)

  def test_incomplete_edge_day_dropped(self, tmp_path, spec):
    # 10 steps: days 0 and 1 complete, day 2 has only two steps -> dropped.
    rng = np.random.default_rng(2)
    steps = [[make_step(rng, 1) for _ in range(10)]]
    ds = run_writer(tmp_path, spec, 10, [(0,)], steps, 1)
    assert ds.sizes["prediction_timedelta_daily"] == 2
    assert ds.sizes["prediction_timedelta"] == 10
    assert np.isfinite(ds["t_850"].values).all()

  def test_no_complete_days(self, tmp_path, spec):
    rng = np.random.default_rng(3)
    steps = [[make_step(rng, 1) for _ in range(2)]]
    ds = run_writer(tmp_path, spec, 2, [(0,)], steps, 1)
    assert ds.sizes["prediction_timedelta_daily"] == 0
    assert np.isfinite(ds["2t"].values).all()

  def test_deterministic_mode_has_no_number_dim(self, tmp_path, spec):
    rng = np.random.default_rng(4)
    steps = [[make_step(rng, None) for _ in range(4)]]
    ds = run_writer(tmp_path, spec, 4, [None], steps, None)
    assert "number" not in ds.dims
    assert ds["2t"].dims == ("time", "prediction_timedelta", "lat", "lon")

  def test_keepbits_rounds_mantissa(self, tmp_path):
    config = write_config(tmp_path, """
        variables:
          2t: {units: K, aggregations: [native]}
        encoding:
          chunks: {prediction_timedelta: 4, lat: 7, lon: 6}
          shards: {prediction_timedelta: 4, lat: 7, lon: 6}
          keepbits: 7
        """)
    spec = zarr_config.load(config)
    rng = np.random.default_rng(5)
    steps = [[make_step(rng, 1) for _ in range(4)]]
    ds = run_writer(tmp_path, spec, 4, [(0,)], steps, 1)
    got = ds["2t"].isel(time=0, number=0).values
    want = np.stack([s["2m_temperature"].values[0] for s in steps[0]])
    assert not np.array_equal(got, want)          # lossy...
    np.testing.assert_allclose(got, want, rtol=2**-7)  # ...but bounded.

  # ------------------------- failure paths -------------------------- #

  def test_worker_error_reraised_and_partial_left(self, tmp_path, spec):
    rng = np.random.default_rng(6)
    path = tmp_path / "out.zarr"
    writer = zarr_stream.ZarrForecastWriter(
        str(path), spec, init_time=INIT, timestep=STEP, n_steps=4,
        lat=LAT_DESC, lon=LON, n_members=1)
    writer.start_members([0])
    bad = make_step(rng, 1)
    bad["lat"] = ("lat", np.arange(7, dtype=np.float32))  # wrong grid
    writer.submit(bad)
    writer._worker.join(timeout=30)  # The worker dies on the bad step.
    with pytest.raises(RuntimeError, match="worker failed"):
      writer.submit(make_step(rng, 1))
    with pytest.raises(RuntimeError, match="worker failed"):
      writer.close()
    assert (tmp_path / "out_partial.zarr").exists()
    assert not path.exists()

  def test_missing_variable_raises_at_submit(self, tmp_path, spec):
    writer = zarr_stream.ZarrForecastWriter(
        str(tmp_path / "out.zarr"), spec, init_time=INIT, timestep=STEP,
        n_steps=4, lat=LAT_DESC, lon=LON, n_members=1)
    writer.start_members([0])
    step = make_step(np.random.default_rng(7), 1).drop_vars("geopotential")
    with pytest.raises(KeyError):
      writer.submit(step)
    writer.abort()

  def test_start_members_rejections(self, tmp_path, spec):
    rng = np.random.default_rng(8)
    writer = zarr_stream.ZarrForecastWriter(
        str(tmp_path / "out.zarr"), spec, init_time=INIT, timestep=STEP,
        n_steps=2, lat=LAT_DESC, lon=LON, n_members=4)
    with pytest.raises(ValueError, match="expected the next group"):
      writer.start_members([1, 2])          # must start at 0
    writer.start_members([0, 1])
    writer.submit(make_step(rng, 2))
    with pytest.raises(ValueError, match="short"):
      writer.start_members([2, 3])          # previous group incomplete
    writer.submit(make_step(rng, 2))
    with pytest.raises(ValueError, match="out of order"):
      writer.start_members([0, 1])          # repeat
    with pytest.raises(ValueError, match="not contiguous"):
      writer.start_members([3, 2])
    with pytest.raises(ValueError, match="out of range"):
      writer.start_members([2, 3, 4])
    with pytest.raises(ValueError, match="never started"):
      writer.close()                        # members 2-3 missing
    writer.abort()

  def test_too_many_submits_rejected(self, tmp_path, spec):
    rng = np.random.default_rng(9)
    writer = zarr_stream.ZarrForecastWriter(
        str(tmp_path / "out.zarr"), spec, init_time=INIT, timestep=STEP,
        n_steps=1, lat=LAT_DESC, lon=LON, n_members=1)
    writer.start_members([0])
    writer.submit(make_step(rng, 1))
    with pytest.raises(ValueError, match="already submitted"):
      writer.submit(make_step(rng, 1))
    writer.abort()

  def test_abort_on_exception_leaves_partial(self, tmp_path, spec):
    rng = np.random.default_rng(10)
    path = tmp_path / "out.zarr"
    with pytest.raises(RuntimeError, match="boom"):
      with zarr_stream.ZarrForecastWriter(
          str(path), spec, init_time=INIT, timestep=STEP, n_steps=4,
          lat=LAT_DESC, lon=LON, n_members=1) as writer:
        writer.start_members([0])
        writer.submit(make_step(rng, 1))
        raise RuntimeError("boom")
    assert (tmp_path / "out_partial.zarr").exists()
    assert not path.exists()

  def test_existing_store_refused(self, tmp_path, spec):
    path = tmp_path / "out.zarr"
    path.mkdir()
    with pytest.raises(FileExistsError):
      zarr_stream.ZarrForecastWriter(
          str(path), spec, init_time=INIT, timestep=STEP, n_steps=4,
          lat=LAT_DESC, lon=LON, n_members=1)

  def test_chunk_and_shard_layout(self, tmp_path, spec):
    import zarr as zarr_lib
    rng = np.random.default_rng(11)
    steps = [[make_step(rng, 1) for _ in range(4)]]
    run_writer(tmp_path, spec, 4, [(0,)], steps, 1)
    group = zarr_lib.open_group(str(tmp_path / "out.zarr"))
    arr = group["2t"]
    assert arr.shards == (1, 1, 10, 6, 6)
    assert arr.chunks == (1, 1, 5, 3, 6)

  def test_shard_files_are_append_only_with_valid_index(self, tmp_path, spec):
    """Parses shard binaries directly: monotone offsets prove append-only."""
    import crc32c
    rng = np.random.default_rng(12)
    n_steps = 12  # chunk 5, shard 10: shard 0 spans two flushes.
    steps = [[make_step(rng, 2) for _ in range(n_steps)]]
    run_writer(tmp_path, spec, n_steps, [(0, 1)], steps, 2)

    shard_files = sorted((tmp_path / "out.zarr" / "2t").rglob("c/*/*/*/*/*"))
    # 2 members x 2 pt-shards (chunks 0-1, edge chunk 2) x 2 lat-shards.
    assert len(shard_files) == 8
    # cps: time 1, number 1, pt 10/5=2, lat 6/3=2, lon 6/6=1 -> 4 entries.
    index_size = 4 * 2 * 8
    for path in shard_files:
      blob = path.read_bytes()
      index_bytes = blob[-(index_size + 4):-4]
      assert blob[-4:] == crc32c.crc32c(index_bytes).to_bytes(4, "little")
      entries = np.frombuffer(index_bytes, dtype="<u8").reshape(-1, 2)
      present = entries[entries[:, 0] != 2**64 - 1]
      assert len(present) >= 1
      # Offsets strictly increasing in file order and tightly packed:
      # exactly what appending produces.
      order = np.argsort(present[:, 0])
      packed = present[order]
      assert packed[0, 0] == 0
      assert (packed[1:, 0] == packed[:-1, 0] + packed[:-1, 1]).all()
      assert packed[-1, 0] + packed[-1, 1] == len(blob) - index_size - 4
