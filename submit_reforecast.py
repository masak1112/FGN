#!/usr/bin/env python3
"""Plans and submits the WeatherNext 2 reforecast as a Slurm job array.

Reads reforecast_config.yaml, works out which init dates still need running,
splits them evenly across `resources.n_jobs` array tasks, and writes a self
contained run directory holding the date lists, a config snapshot and the batch
script.

Nothing is created, deleted or submitted unless --submit is passed: by default
this only surveys the archive and prints the plan, including the generated batch
script.

  python submit_reforecast.py                    # plan only
  python submit_reforecast.py --n_jobs 40        # plan with a different split
  python submit_reforecast.py --submit           # write the run dir and sbatch

Each array task runs wn2_inference.py over its own date list in a single
process, so the ~3 minute model compilation is paid once per job rather than
once per date. Logs land under

  <log_dir>/<run>/slurm/task_<a>.job_<A>.out      one file per array task
  <log_dir>/<run>/tasks/task_<a>/wn2.log          that task's whole run
  <log_dir>/<run>/tasks/task_<a>/<init>/perf.log  per-init timings

A run directory is a claim on its dates, and two things follow from that. An
array task reads its date list and the config snapshot when Slurm *starts* it,
not when it was submitted, so an existing run directory is never written over.
And inits that another run has queued against the same store directory are left
out of this submission: two jobs on one init would both stage into the same
`_partial.zarr`, where the second to start deletes the first's work in progress.
"""

import argparse
import dataclasses
import datetime
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import pandas as pd
import yaml

import reforecast_config

# Measured on 4x h200 (o001) at 1200h / 32 members: 3.73 s per pmapped group of
# `gpu_per_job` members advancing one 6-hourly step, averaged over the run.
SECONDS_PER_STEP_GROUP = 3.73
# One-time per-process cost: interpreter start, fiddle config, JIT compilation.
STARTUP_SECONDS = 260.0
# Per-init cost outside the rollout: ERA5 input load, store preallocation and
# the final flush of buffered shards.
PER_INIT_OVERHEAD_SECONDS = 80.0
# Fraction of the time limit the suggested dates-per-job aims to fill. The
# estimate carries real variance — ERA5 reads come over the network and node
# speeds differ — and overshooting costs a whole init's work.
SAFETY_FRACTION = 0.9
# Fraction of the uncompressed float32 volume that lands on disk: 164 GiB of a
# 285 GiB store, measured with `encoding.keepbits: null`. Setting keepbits cuts
# this substantially, so treat the estimate as an upper bound.
COMPRESSION_RATIO = 0.58

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_DIR / "reforecast_config.yaml"),
        help="Reforecast config. Default: %(default)s.",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Write the run directory and submit the array. Without this the "
        "script only reads the archive and prints the plan.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Override resources.n_jobs for this submission.",
    )
    p.add_argument(
        "--n_concurrent_jobs",
        type=int,
        default=None,
        help="Override resources.n_concurrent_jobs (the array %% throttle).",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="Override reforecast.year_range for this submission.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Submit at most this many pending inits (earliest first). Useful "
        "for a first slice of a campaign that does not fit on disk.",
    )
    p.add_argument(
        "--interleave",
        action="store_true",
        help="Deal dates round-robin instead of in contiguous blocks, so a "
        "campaign that stops early still covers every year evenly.",
    )
    p.add_argument(
        "--ignore_queued",
        action="store_true",
        help="Include inits that another run directory has already queued. Two "
        "jobs on one init race on the same _partial store, so pass this only "
        "when the other array is known to be dead (scancel'd, or its run dir "
        "left over from a previous cluster).",
    )
    p.add_argument(
        "--clean_partial",
        action="store_true",
        help="With --submit, delete staged `_partial.zarr` stores from failed "
        "runs to reclaim disk. They are rewritten from scratch either way.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Submit even when a job's estimated runtime exceeds the time "
        "limit, which otherwise refuses: such jobs are killed mid-init and the "
        "unfinished date's work is lost.",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Name of the run directory under the config's log_dir. "
        "Default: run_<UTC timestamp>.",
    )
    p.add_argument(
        "--time_limit",
        default=None,
        help="Override resources.time_limit for this submission.",
    )
    return p.parse_args()


def replace_nested(cfg, section: str, **changes):
    """Returns `cfg` with one section's fields replaced."""
    return dataclasses.replace(
        cfg, **{section: dataclasses.replace(getattr(cfg, section), **changes)}
    )


def under_project(path) -> pathlib.Path:
    """A configured path made absolute.

    Relative paths are project-relative, which is what they resolve to at
    runtime anyway: the batch script cds to PROJECT_DIR before doing anything.
    """
    path = pathlib.Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def free_bytes(path: pathlib.Path) -> int:
    """Free space on the filesystem that will hold `path`, which may not exist."""
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return shutil.disk_usage(candidate).free
    return 0


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} PiB"


def human_hours(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"


def survey(cfg, inits):
    """Splits the campaign's inits by what is already on disk.

    A published store means the init is done: zarr_stream only renames its
    staging directory into place after the completeness checks in close(), so a
    store existing at the final path is a complete store. A leftover
    `_partial.zarr` is a failed or killed run; it is deleted and rewritten when
    the init runs again.
    """
    done, staged, pending = [], [], []
    for init in inits:
        if cfg.store_path(init).exists():
            done.append(init)
        else:
            if cfg.staging_path(init).exists():
                staged.append(init)
            pending.append(init)
    return done, staged, pending


def queued_array_tasks() -> dict[int, set[int]] | None:
    """Array task indices Slurm still holds, keyed by base array job id.

    The whole queue is read rather than one user's: run directories are
    group-shared, so a colleague's submission can collide with ours. `--array`
    puts every array element on its own line, which keeps %K a single index
    instead of a range like `0-39%4`. Returns None when squeue cannot be
    reached, so the caller can say so rather than quietly assume an empty queue.
    """
    try:
        result = subprocess.run(
            # %F is the base array job id, the number sbatch prints; %A is the
            # element's own id, which Slurm reassigns when the element starts.
            ["squeue", "--array", "--noheader", "--format=%F|%K"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    tasks: dict[int, set[int]] = {}
    for line in result.stdout.splitlines():
        job, _, task = line.partition("|")
        # Plain jobs report %K as N/A; a run of ours is always an array, even
        # at n_jobs=1, so skipping non-numeric indices only drops other people's
        # non-array work.
        if job.strip().isdigit() and task.strip().isdigit():
            tasks.setdefault(int(job), set()).add(int(task))
    return tasks


def run_job_id(run_dir: pathlib.Path) -> int | None:
    """The array job id a run was submitted as, or None if it never was."""
    try:
        text = (run_dir / "sbatch.out").read_text()
    except OSError:
        return None
    match = re.search(r"Submitted batch job (\d+)", text)
    return int(match.group(1)) if match else None


def run_store_dir(run_dir: pathlib.Path) -> pathlib.Path | None:
    """Where a run's config snapshot publishes to, or None if unreadable.

    The snapshot is read as plain YAML rather than through reforecast_config so
    that an older or newer schema still yields the one key that matters here.
    """
    try:
        with open(run_dir / "config.yaml") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return None
    data = raw.get("data") or {}
    default = reforecast_config.Data().data_dir
    return under_project(data.get("data_dir") or default).resolve()


def read_dates_file(path: pathlib.Path) -> list[pd.Timestamp]:
    """Parses one task's date list, as wn2_inference's --init_file does."""
    lines = [line.partition("#")[0].strip() for line in path.read_text().splitlines()]
    return [pd.Timestamp(line) for line in lines if line]


def claimed_inits(cfg, log_dir: pathlib.Path, queued) -> dict[str, set]:
    """Inits other live runs have claimed, keyed by run directory name.

    Only tasks Slurm still knows about count. A task that has finished either
    published its stores, which `survey` already sees, or died, which leaves its
    dates genuinely pending — either way its date list is no longer a claim.
    """
    ours = under_project(cfg.data.data_dir).resolve()
    claims: dict[str, set] = {}
    if not log_dir.is_dir():
        return claims
    for run_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        job_id = run_job_id(run_dir)
        if job_id is None or job_id not in queued:
            continue
        # An unparseable snapshot is treated as a conflict: better to leave a
        # date to a run that may not want it than to double-submit it.
        store_dir = run_store_dir(run_dir)
        if store_dir is not None and store_dir != ours:
            continue
        for task in sorted(queued[job_id]):
            dates_file = run_dir / "dates" / f"task_{task:03d}.txt"
            try:
                inits = read_dates_file(dates_file)
            except (OSError, ValueError):
                continue  # Not one of our run dirs, or a hand-edited list.
            if inits:
                claims.setdefault(run_dir.name, set()).update(inits)
    return claims


def unique_run_dir(log_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    """The first free `<run_name>`, `<run_name>_2`, `<run_name>_3`, ... ."""
    candidate = log_dir / run_name
    suffix = 2
    while candidate.exists():
        candidate = log_dir / f"{run_name}_{suffix}"
        suffix += 1
    return candidate


def directory_bytes(path: pathlib.Path) -> int:
    total = 0
    for root, _, files in os.walk(path, onerror=lambda _: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def split_dates(inits, n_jobs: int, interleave: bool) -> list[list]:
    """Deals inits into `n_jobs` groups whose sizes differ by at most one."""
    groups: list[list] = [[] for _ in range(n_jobs)]
    if interleave:
        for i, init in enumerate(inits):
            groups[i % n_jobs].append(init)
        return groups
    # Contiguous blocks: the first `remainder` groups take one extra date.
    per_job, remainder = divmod(len(inits), n_jobs)
    start = 0
    for i in range(n_jobs):
        size = per_job + (1 if i < remainder else 0)
        groups[i] = list(inits[start : start + size])
        start += size
    return groups


def estimate_seconds(cfg, n_dates: int) -> float:
    """Wall-clock estimate for one array task running `n_dates` inits."""
    r, f = cfg.resources, cfg.reforecast
    step_groups = f.n_steps * f.n_members / r.gpu_per_job
    per_date = step_groups * SECONDS_PER_STEP_GROUP + PER_INIT_OVERHEAD_SECONDS
    return STARTUP_SECONDS + n_dates * per_date


def build_sbatch(cfg, run_dir: pathlib.Path, n_jobs: int, concurrent, time_limit):
    """Renders the array script. `run_dir` paths are absolute so cwd cannot bite."""
    r = cfg.resources
    array = f"0-{n_jobs - 1}" + (f"%{concurrent}" if concurrent else "")
    directives = [
        f"#SBATCH --job-name={r.job_name}",
        f"#SBATCH --partition={r.partition}",
        f"#SBATCH --array={array}",
        f"#SBATCH --nodes={r.nodes_per_job}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={r.cpu_per_task}",
        f"#SBATCH --gres={r.gres}",
        f"#SBATCH --mem={r.mem_per_job}",
        f"#SBATCH --time={time_limit}",
        # stderr is merged into this file; %A is the array job id, %a the task.
        f"#SBATCH --output={run_dir}/slurm/task_%a.job_%A.out",
        "#SBATCH --open-mode=append",
    ]
    if r.exclude_nodes:
        directives.append(f"#SBATCH --exclude={r.exclude_nodes}")

    umask = (
        f"umask {reforecast_config.GROUP_UMASK:03o}\n"
        if cfg.data.group_permissions
        else ""
    )
    body = textwrap.dedent(
        f"""
        set -euo pipefail

        cd {PROJECT_DIR}
        source ./.venv/bin/activate
        {umask}
        task=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
        dates_file={run_dir}/dates/task_${{task}}.txt
        task_log_dir={run_dir}/tasks/task_${{task}}

        echo "host        : $(hostname)"
        echo "job / task  : ${{SLURM_ARRAY_JOB_ID}} / ${{SLURM_ARRAY_TASK_ID}}"
        echo "dates       : $dates_file ($(grep -Ecv '^\\s*(#|$)' "$dates_file") inits)"
        echo "logs        : $task_log_dir"
        echo "started     : $(date -Is)"
        nvidia-smi
        echo

        python -u wn2_inference.py \\
            --config {run_dir}/config.yaml \\
            --init_file "$dates_file" \\
            --log_dir "$task_log_dir"

        echo "finished    : $(date -Is)"
        """
    ).strip()
    return "#!/bin/bash -l\n" + "\n".join(directives) + "\n\n" + body + "\n"


def write_run_dir(cfg, run_dir, groups, sbatch_text) -> pathlib.Path:
    """Materializes the run directory. Only called under --submit."""
    run_dir = pathlib.Path(run_dir)
    if run_dir.exists():
        # main() refuses before reaching here; this is the backstop, because
        # overwriting a run dir redirects any of its tasks still in the queue.
        raise FileExistsError(
            f"{run_dir} already exists; refusing to overwrite a run's date "
            "lists or config snapshot."
        )
    group_perms = cfg.data.group_permissions
    reforecast_config.make_dir(run_dir, group_perms)
    for sub in ("dates", "slurm", "tasks"):
        reforecast_config.make_dir(run_dir / sub, group_perms)

    # Snapshot the config so a resubmitted or inspected run is reproducible
    # even after reforecast_config.yaml moves on.
    shutil.copyfile(cfg.path, run_dir / "config.yaml")
    for i, dates in enumerate(groups):
        lines = [f"# task {i:03d}: {len(dates)} init date(s)"]
        lines += [pd.Timestamp(d).strftime("%Y-%m-%dT%H") for d in dates]
        (run_dir / "dates" / f"task_{i:03d}.txt").write_text("\n".join(lines) + "\n")

    script = run_dir / "submit.sbatch"
    script.write_text(sbatch_text)
    # Group-writable too, so anyone in the group can tweak and resubmit a run.
    script.chmod(0o770 if group_perms else 0o755)
    return script


def main():
    args = parse_args()
    cfg = reforecast_config.load(args.config)
    if cfg.data.group_permissions:
        # Set before anything is written, and before sbatch: Slurm hands the
        # submitter's umask to the job, which is what makes the per-task stdout
        # files group-writable.
        reforecast_config.apply_group_permissions()
    if args.years:
        cfg = replace_nested(cfg, "reforecast", year_range=tuple(args.years))
    n_jobs = args.n_jobs or cfg.resources.n_jobs
    concurrent = args.n_concurrent_jobs or cfg.resources.n_concurrent_jobs
    time_limit = args.time_limit or cfg.resources.time_limit
    if args.time_limit:
        cfg = replace_nested(cfg, "resources", time_limit=time_limit)
    if n_jobs < 1:
        sys.exit("n_jobs must be >= 1.")

    r, f = cfg.resources, cfg.reforecast
    inits = cfg.init_times()
    data_dir = pathlib.Path(cfg.data.data_dir)
    log_dir = under_project(r.log_dir)
    done, staged, pending = survey(cfg, inits)

    # What another run has already taken. Done before --limit so the limit
    # counts inits this submission will really compute.
    queued_tasks = None if args.ignore_queued else queued_array_tasks()
    claims = claimed_inits(cfg, log_dir, queued_tasks) if queued_tasks else {}
    claimed = set().union(*claims.values()) if claims else set()
    claimed &= set(pending)
    pending = [i for i in pending if i not in claimed]
    # A staged store whose init another run is working on is not debris from a
    # failed run: it is being written right now, and must not be cleaned.
    in_flight = [i for i in staged if i in claimed]
    staged = [i for i in staged if i not in claimed]

    print(f"FGN reforecast plan — {cfg.path}")
    print()
    print(f"  store dir     {data_dir}" + ("" if data_dir.exists() else "  (new)"))
    print(
        f"  forecast      {f.lead_hours}h ({f.n_steps} steps, {f.n_daily} daily), "
        f"{f.n_members} members from {f.n_checkpoints} checkpoints"
    )
    print(
        f"  init dates    {len(inits)}  "
        f"({f.year_range[0]}-{f.year_range[1]}, 00Z"
        + (
            f", excluding months {list(f.exclude_months)}"
            if f.exclude_months
            else ""
        )
        + ")"
    )
    print(f"    published   {len(done)}")
    if args.ignore_queued:
        print("    queued      not checked  (--ignore_queued)")
    elif queued_tasks is None:
        print(
            "    queued      unknown  <<< squeue failed, so inits another run "
            "has already claimed cannot be excluded"
        )
    else:
        print(
            f"    queued      {len(claimed)}"
            + (f"  (claimed by {', '.join(sorted(claims))})" if claims else "")
        )
    staged_bytes = sum(directory_bytes(cfg.staging_path(i)) for i in staged)
    notes = []
    if staged:
        notes.append(f"{human_bytes(staged_bytes)} in _partial stores from failed runs")
    if in_flight:
        notes.append(f"{len(in_flight)} being written by a queued run")
    print(f"    staged      {len(staged)}" + (f"  ({'; '.join(notes)})" if notes else ""))
    print(f"    pending     {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"  --limit       first {len(pending)} pending init(s)")
    if not pending:
        print(
            "\nNothing to submit: every init in the config is already published"
            + (" or queued by another run." if claimed else ".")
        )
        return 0

    n_jobs = min(n_jobs, len(pending))
    groups = [g for g in split_dates(pending, n_jobs, args.interleave) if g]
    n_jobs = len(groups)
    sizes = sorted({len(g) for g in groups})
    per_job = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
    est = max(estimate_seconds(cfg, len(g)) for g in groups)
    limit = r.time_limit_seconds

    print()
    print(
        f"  jobs          {n_jobs} array task(s)"
        + (f", at most {concurrent} at a time" if concurrent else "")
        + (", interleaved" if args.interleave else ", contiguous blocks")
    )
    print(
        f"  resources     {r.nodes_per_job} node, {r.gres}, "
        f"{r.cpu_per_task} cpus, {r.mem_per_job}, partition {r.partition}"
        + (f", excluding {r.exclude_nodes}" if r.exclude_nodes else "")
    )
    print(f"  dates/job     {per_job}")
    print(
        f"  est. runtime  {human_hours(est)} per job "
        f"vs the {r.time_limit} limit"
        + ("  <<< OVER THE LIMIT" if est > limit else "")
    )
    if est > limit:
        per_date = estimate_seconds(cfg, 2) - estimate_seconds(cfg, 1)
        fits = max(1, int((limit * SAFETY_FRACTION - STARTUP_SECONDS) // per_date))
        print(
            f"                use --n_jobs {-(-len(pending) // fits)} or more "
            f"({fits} date(s) per job, {human_hours(estimate_seconds(cfg, fits))}, "
            f"keeps {round((1 - SAFETY_FRACTION) * 100)}% headroom)"
        )

    store_bytes = cfg.store_bytes()
    on_disk = store_bytes * COMPRESSION_RATIO
    total = on_disk * len(pending)
    free = free_bytes(data_dir)
    print(
        f"  storage       ~{human_bytes(on_disk)} per init "
        f"({human_bytes(store_bytes)} uncompressed x {COMPRESSION_RATIO:.2f}), "
        f"~{human_bytes(total)} for {len(pending)} init(s)"
    )
    print(
        f"                {human_bytes(free)} free on {data_dir}"
        + ("  <<< NOT ENOUGH SPACE" if total > free else "")
    )
    if total > free:
        print(
            f"                {int(free // on_disk)} init(s) fit; use --limit, "
            "narrow --years, or set encoding.keepbits to shrink each store"
        )

    if args.run_name:
        # An explicit name that is taken is refused rather than moved aside: the
        # caller asked for that directory, and its date lists may still be
        # feeding an array that Slurm has not started yet.
        run_dir = log_dir / args.run_name
        name_taken = run_dir.exists()
    else:
        # The default name is a UTC timestamp, so a clash means two submissions
        # in one second. That is an accident, not an instruction — take the next
        # free name.
        run_dir = unique_run_dir(
            log_dir, "run_" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        name_taken = False
    sbatch_text = build_sbatch(cfg, run_dir, n_jobs, concurrent, time_limit)

    print()
    print(f"  run dir       {run_dir}" + ("  <<< ALREADY EXISTS" if name_taken else ""))
    if staged and args.clean_partial:
        verb = "deleting" if args.submit else "would delete"
        print(
            f"  --clean_partial {verb} {len(staged)} staged store(s), "
            f"reclaiming {human_bytes(staged_bytes)}:"
        )
        for init in staged:
            print(f"                  {cfg.staging_path(init)}")
    elif staged:
        print(
            f"  note          {len(staged)} staged store(s) will be rewritten from "
            "scratch; pass --clean_partial to reclaim their space first"
        )
    if in_flight and args.clean_partial:
        print(
            f"  keeping       {len(in_flight)} staged store(s) whose init is queued "
            "by another run: deleting those would destroy work in progress"
        )

    if not args.submit:
        print()
        print("  Generated batch script (nothing has been written):")
        print()
        print(textwrap.indent(sbatch_text, "    "))
        print("  Re-run with --submit to write the run directory and queue the array.")
        return 0

    if name_taken:
        print()
        return (
            f"Refusing to submit: {run_dir} already exists. Its array tasks read "
            f"dates/task_<a>.txt and config.yaml when Slurm starts them, so "
            f"overwriting that directory would redirect any task of that run "
            f"still in the queue. Choose another --run_name, or drop --run_name "
            f"for a timestamped one."
        )

    if est > limit and not args.force:
        print()
        return (
            f"Refusing to submit: each job needs about {human_hours(est)} but the "
            f"limit is {r.time_limit}, so every task would be killed part-way "
            f"through an init and that date's work lost. Raise --n_jobs, lower "
            f"--limit, or pass --force to submit anyway."
        )

    if args.clean_partial:
        for init in staged:
            shutil.rmtree(cfg.staging_path(init), ignore_errors=True)

    reforecast_config.make_dir(data_dir, cfg.data.group_permissions)
    script = write_run_dir(cfg, run_dir, groups, sbatch_text)
    print(f"  wrote         {script}")

    result = subprocess.run(
        ["sbatch", str(script)], capture_output=True, text=True, check=False
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    (run_dir / "sbatch.out").write_text(result.stdout)
    print(f"  logs          {run_dir}/slurm/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
