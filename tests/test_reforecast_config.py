"""Tests for reforecast_config parsing and submit_reforecast's job planning.

The planning tests cover the parts that fail silently rather than loudly: the
Slurm time format (YAML turns an unquoted 12:00:00 into a number), the even
split of init dates across jobs, and the survey that decides which dates still
need running.
"""

import os
import stat
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

import reforecast_config
import submit_reforecast
import zarr_config
import zarr_stream

CONFIG_PATH = "reforecast_config.yaml"


def write_config(tmp_path, raw) -> str:
  path = tmp_path / "config.yaml"
  path.write_text(yaml.safe_dump(raw, sort_keys=False))
  return str(path)


@pytest.fixture
def raw():
  with open(CONFIG_PATH) as f:
    return yaml.safe_load(f)


# --------------------------------------------------------------------- #
# Config parsing                                                         #
# --------------------------------------------------------------------- #


class TestConfig:

  def test_example_config_parses(self, raw):
    cfg = reforecast_config.load(CONFIG_PATH)
    assert cfg.reforecast.n_checkpoints == len(cfg.reforecast.weights)
    assert cfg.reforecast.n_steps == cfg.reforecast.lead_hours // 6
    # The variables/encoding sections parse exactly as zarr_config would.
    assert cfg.output_spec == zarr_config.parse(raw["variables"], raw["encoding"])

  def test_extra_top_level_sections_are_allowed(self, tmp_path, raw):
    """zarr_config.load rejects these; the reforecast loader owns them."""
    with pytest.raises(ValueError, match="Unknown top-level"):
      zarr_config.load(write_config(tmp_path, raw))
    assert reforecast_config.load(write_config(tmp_path, raw))

  def test_unknown_key_raises(self, tmp_path, raw):
    raw["reforecast"]["lead_hrs"] = 240
    with pytest.raises(ValueError, match="Unknown keys.*'reforecast'"):
      reforecast_config.load(write_config(tmp_path, raw))

  def test_unknown_top_level_key_raises(self, tmp_path, raw):
    raw["resource"] = {}
    with pytest.raises(ValueError, match="Unknown top-level"):
      reforecast_config.load(write_config(tmp_path, raw))

  @pytest.mark.parametrize(
      "text,seconds",
      [("30", 1800), ("5:30", 330), ("12:00:00", 43200), ("2-12", 216000),
       ("1-06:30", 109800), ("3-00:00:00", 259200)],
  )
  def test_time_limit_formats(self, text, seconds):
    assert reforecast_config.Resources(time_limit=text).time_limit_seconds == seconds

  def test_unquoted_time_limit_is_rejected(self, tmp_path, raw):
    """YAML 1.1 reads 12:00:00 as sexagesimal, i.e. the int 43200."""
    assert yaml.safe_load("t: 12:00:00")["t"] == 43200
    raw["resources"]["time_limit"] = 43200
    with pytest.raises(ValueError, match="must be quoted"):
      reforecast_config.load(write_config(tmp_path, raw))

  def test_config_time_limit_is_a_string(self, raw):
    """Guards the config itself against losing its quotes."""
    assert isinstance(raw["resources"]["time_limit"], str)

  def test_gres_pins_the_gpu_type(self):
    assert reforecast_config.Resources(gpu_type="h200", gpu_per_job=4).gres == (
        "gpu:h200:4")
    assert reforecast_config.Resources(gpu_per_job=2).gres == "gpu:2"

  @pytest.mark.parametrize(
      "section,change,match",
      [("reforecast", {"n_members": 30}, "must be a multiple"),
       ("reforecast", {"members_per_checkpoint": 32}, "checkpoints, but"),
       ("reforecast", {"lead_hours": 100}, "multiple of 6"),
       ("reforecast", {"year_range": [2020, 2000]}, "first <= last"),
       ("reforecast", {"exclude_months": [13]}, "invalid month"),
       ("resources", {"n_jobs": 0}, "must be >= 1")],
  )
  def test_validation(self, tmp_path, raw, section, change, match):
    raw[section].update(change)
    with pytest.raises(ValueError, match=match):
      reforecast_config.load(write_config(tmp_path, raw))

  def test_members_must_divide_across_gpus(self, tmp_path, raw):
    raw["resources"]["gpu_per_job"] = 3  # 8 members per checkpoint / 3 GPUs.
    with pytest.raises(ValueError, match="multiple of resources.gpu_per_job"):
      reforecast_config.load(write_config(tmp_path, raw))

  def test_level_checked_against_the_model(self, tmp_path, raw):
    with pytest.raises(ValueError, match=r"not among the model's levels"):
      reforecast_config.load(write_config(tmp_path, raw), available_levels=[500])


class TestInitDates:

  def test_year_range_is_inclusive_and_sorted(self, tmp_path, raw):
    raw["reforecast"]["year_range"] = [2000, 2002]
    inits = reforecast_config.load(write_config(tmp_path, raw)).init_times()
    assert inits == sorted(inits)
    assert {d.year for d in inits} == {2000, 2001, 2002}
    assert {d.hour for d in inits} == {0}
    assert len(inits) == len(set(inits))

  def test_exclude_months(self, tmp_path, raw):
    raw["reforecast"]["year_range"] = [2020, 2020]
    raw["reforecast"]["exclude_months"] = [1, 2, 12]
    inits = reforecast_config.load(write_config(tmp_path, raw)).init_times()
    assert {d.month for d in inits} == {3, 4, 5, 6, 7, 8, 9, 10, 11}

  def test_store_paths_follow_the_archive_convention(self, tmp_path, raw):
    raw["data"]["data_dir"] = str(tmp_path)
    cfg = reforecast_config.load(write_config(tmp_path, raw))
    init = pd.Timestamp("2020-08-31T00")
    assert cfg.store_path(init) == tmp_path / "2020-08-31T00.zarr"
    assert cfg.store_path(init, "nc") == tmp_path / "2020-08-31T00.nc"

  def test_staging_path_matches_the_writer(self, tmp_path, raw):
    """cfg.staging_path must name the same directory the writer stages into.

    The survey treats a `_partial` store as an unfinished run, so the two have
    to agree or failed runs go unnoticed.
    """
    raw["data"]["data_dir"] = str(tmp_path)
    cfg = reforecast_config.load(write_config(tmp_path, raw))
    init = pd.Timestamp("2020-08-31T00")
    spec = zarr_config.parse(
        {"2t": {"units": "K", "aggregations": ["native"]}},
        {"chunks": {"prediction_timedelta": 2, "lat": 3, "lon": 6},
         "shards": {"prediction_timedelta": 2, "lat": 3, "lon": 6}},
    )
    writer = zarr_stream.ZarrForecastWriter(
        str(cfg.store_path(init)), spec, init_time=init,
        timestep=pd.Timedelta("6h"), n_steps=2,
        lat=np.linspace(90, -90, 3, dtype=np.float32),
        lon=np.arange(0, 360, 60, dtype=np.float32), n_members=1)
    try:
      assert cfg.staging_path(init).exists()
      assert not cfg.store_path(init).exists()
    finally:
      writer.abort()


# --------------------------------------------------------------------- #
# Job planning                                                           #
# --------------------------------------------------------------------- #


class TestPlanning:

  @pytest.mark.parametrize("n_dates,n_jobs", [(2366, 40), (91, 13), (10, 10),
                                              (7, 3), (1, 1)])
  @pytest.mark.parametrize("interleave", [False, True])
  def test_split_is_even_and_lossless(self, n_dates, n_jobs, interleave):
    dates = list(range(n_dates))
    groups = submit_reforecast.split_dates(dates, n_jobs, interleave)
    assert len(groups) == n_jobs
    assert sorted(d for g in groups for d in g) == dates  # No loss, no repeats.
    sizes = [len(g) for g in groups]
    assert max(sizes) - min(sizes) <= 1

  def test_contiguous_split_keeps_order(self):
    groups = submit_reforecast.split_dates(list(range(10)), 3, interleave=False)
    assert groups == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

  def test_interleaved_split_spreads_years(self):
    groups = submit_reforecast.split_dates(list(range(10)), 3, interleave=True)
    assert groups == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]

  def test_more_jobs_than_dates_leaves_empty_groups(self):
    groups = submit_reforecast.split_dates([1, 2], 5, interleave=False)
    assert [g for g in groups if g] == [[1], [2]]

  def test_survey_classifies_stores(self, tmp_path, raw):
    raw["data"]["data_dir"] = str(tmp_path)
    raw["reforecast"]["year_range"] = [2020, 2020]
    cfg = reforecast_config.load(write_config(tmp_path, raw))
    inits = cfg.init_times()
    cfg.store_path(inits[0]).mkdir()          # Published.
    cfg.staging_path(inits[1]).mkdir()        # Killed mid-run.
    done, staged, pending = submit_reforecast.survey(cfg, inits)
    assert done == [inits[0]]
    assert staged == [inits[1]]
    # A staged store is unfinished, so its init is still pending.
    assert pending == inits[1:]

  def test_runtime_estimate_scales_with_dates(self):
    cfg = reforecast_config.load(CONFIG_PATH)
    one = submit_reforecast.estimate_seconds(cfg, 1)
    two = submit_reforecast.estimate_seconds(cfg, 2)
    assert one > submit_reforecast.STARTUP_SECONDS
    assert two - one == pytest.approx(one - submit_reforecast.STARTUP_SECONDS)

  def test_store_bytes_matches_the_arrays(self):
    cfg = reforecast_config.load(CONFIG_PATH)
    f, spec = cfg.reforecast, cfg.output_spec
    field = f.n_members * reforecast_config.LAT_SIZE * reforecast_config.LON_SIZE * 4
    assert cfg.store_bytes() == field * (
        len(spec.native_outputs) * f.n_steps + len(spec.daily_outputs) * f.n_daily)


class TestSbatchScript:

  @pytest.fixture
  def script(self, tmp_path):
    cfg = reforecast_config.load(CONFIG_PATH)
    return submit_reforecast.build_sbatch(
        cfg, tmp_path / "run", n_jobs=13, concurrent=4, time_limit="12:00:00")

  def test_directives(self, script):
    assert "#SBATCH --array=0-12%4" in script
    assert "#SBATCH --gres=gpu:h200:4" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --exclude=p001,p003" in script
    assert "#SBATCH --partition=general" in script

  def test_no_throttle_leaves_the_array_open(self, tmp_path):
    cfg = reforecast_config.load(CONFIG_PATH)
    script = submit_reforecast.build_sbatch(
        cfg, tmp_path / "run", n_jobs=3, concurrent=None, time_limit="1:00:00")
    assert "#SBATCH --array=0-2\n" in script

  def test_task_reads_its_own_date_list(self, script):
    assert 'dates_file=' in script and 'task_${task}.txt' in script
    assert '--init_file "$dates_file"' in script
    assert "umask 007" in script  # group_permissions: true

  def test_is_valid_bash(self, tmp_path, script):
    path = tmp_path / "submit.sbatch"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


class TestGroupPermissions:
  """`group_permissions: true` must hold whatever umask the caller starts with.

  Group *ownership* is not tested here: new files take the creating process's
  primary group unless the parent is setgid, which is a property of the
  filesystem (every directory under /net/monsoon is) rather than of this code.
  What these tests pin down is the mode bits, which is the half the code owns.
  """

  @pytest.fixture
  def hostile_umask(self):
    """A umask that would otherwise leave everything group-read-only."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)

  def test_every_created_dir_is_group_rwx_and_setgid(self, tmp_path, hostile_umask):
    reforecast_config.apply_group_permissions()
    leaf = reforecast_config.make_dir(tmp_path / "a" / "b" / "c", True)
    # Not just the leaf: mkdir(parents=True) applies only the umask to the
    # intermediates, so each one has to be chmod'ed too.
    for path in (tmp_path / "a", tmp_path / "a" / "b", leaf):
      assert stat.S_IMODE(path.stat().st_mode) == reforecast_config.GROUP_DIR_MODE

  def test_files_are_group_rw_not_world_readable(self, tmp_path, hostile_umask):
    reforecast_config.apply_group_permissions()
    path = reforecast_config.make_dir(tmp_path / "logs", True) / "wn2.log"
    path.write_text("x")
    assert stat.S_IMODE(path.stat().st_mode) == 0o660

  def test_run_dir_contents(self, tmp_path, raw, hostile_umask):
    reforecast_config.apply_group_permissions()
    cfg = reforecast_config.load(write_config(tmp_path, raw))
    run_dir = tmp_path / "run"
    script = submit_reforecast.write_run_dir(
        cfg, run_dir, [[pd.Timestamp("2020-01-01T00")]], "#!/bin/bash\n")
    assert stat.S_IMODE(script.stat().st_mode) == 0o770  # Executable, group-writable.
    for name in ("config.yaml", "dates/task_000.txt"):
      assert stat.S_IMODE((run_dir / name).stat().st_mode) == 0o660
    for name in ("dates", "slurm", "tasks"):
      assert stat.S_IMODE((run_dir / name).stat().st_mode) == (
          reforecast_config.GROUP_DIR_MODE)

  def test_disabled_leaves_the_umask_alone(self, tmp_path, hostile_umask):
    path = reforecast_config.make_dir(tmp_path / "plain", group_permissions=False)
    assert stat.S_IMODE(path.stat().st_mode) == 0o755  # 0777 & ~022


def test_date_file_round_trip(tmp_path, raw):
  """What write_run_dir emits is what wn2_inference's --init_file parses."""
  raw["reforecast"]["year_range"] = [2020, 2020]
  cfg = reforecast_config.load(write_config(tmp_path, raw))
  inits = cfg.init_times()[:5]
  run_dir = tmp_path / "run"
  submit_reforecast.write_run_dir(cfg, run_dir, [inits], "#!/bin/bash\n")

  text = (run_dir / "dates" / "task_000.txt").read_text()
  lines = [line.partition("#")[0].strip() for line in text.splitlines()]
  assert [pd.Timestamp(x) for x in lines if x] == inits
  # The snapshot is a usable config in its own right.
  assert reforecast_config.load(str(run_dir / "config.yaml")).init_times()
