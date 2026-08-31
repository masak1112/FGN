# FGN Reforecast Campaign Tutorial (2000–2025)

This tutorial explains how to run the WeatherNext 2 (FGN) reforecast for years
2000–2025 using **6 users**, each responsible for a sub-range of years.

---

## Overview

| Item | Value |
|---|---|
| Year range | 2000–2025 (26 years) |
| Init dates | 2,366 total (~91 per year) |
| Forecast length | 1,200 h (50 days), 32 ensemble members |
| Time per init date | ~2 hours (1 node, 4 H100 GPUs) |
| Job time limit | 24 h |
| Dates per job | ~10 (at 90% safety margin) |
| Jobs per user | 3 (cluster limit) |
| Dates per user per wave | 30 |
| Submission cycle | ~24 h |
| **Estimated completion** | **~32 days** |

---

## Year Assignments

Each user is responsible for a contiguous range of years. The script
automatically skips dates that are already done, so resubmitting the same
command is always safe.

| User | Years | Num years | Dates | Waves | Est. days |
|------|-------|-----------|-------|-------|-----------|
| Bo (User 1) | 2000–2003 | 4 | 364 | 13 | ~26 days |
| Rajat (User 2) | 2004–2007 | 4 | 364 | 13 | ~26 days |
| Christ (User 3) | 2008–2011 | 4 | 364 | 13 | ~26 days |
| Anustup (User 4) | 2012–2016 | 5 | 455 | 16 | ~32 days |
| Aryan Kaushal (User 5) | 2017–2020 | 4 | 364 | 13 | ~26 days |
| Bing Gong (User 6) | 2021–2025 | 5 | 455 | 16 | ~32 days |


---

## Step 1 (all users)

1. **SSH into Stampede3** and navigate to the project directory:

   ```bash
   cd /scratch/10786/bgong1/FGN
   ```

## Step 2: Submit Your 3 Jobs

Each user splits their year range into **3 roughly equal sub-ranges** and
submits one job per sub-range. Re-run these same commands every ~24 h —
the script automatically skips already-completed dates.

### Bo (2000–2003)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2000 2001 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2002 2002 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2003 2003 --force --submit
```

### Rajat (2004–2007)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2004 2005 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2006 2006 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2007 2007 --force --submit
```

### Christ (2008–2011)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2008 2009 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2010 2010 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2011 2011 --force --submit
```

### Anustup (2012–2016)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2012 2013 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2014 2015 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2016 2016 --force --submit
```

### Aryan Kaushal (2017–2020)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2017 2018 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2019 2019 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2020 2020 --force --submit
```

### Bing Gong (2021–2025)

```bash
python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2021 2022 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2023 2024 --force --submit

python submit_reforecast.py --config reforecast_config_stampede3.yaml \
    --years 2025 2025 --force --submit
```

---

## Step 3: Monitor Progress

Check your jobs in the queue:

```bash
squeue -u $USER
```

Check how many dates are done vs pending for your year range:

```bash
python submit_reforecast.py \
    --config reforecast_config_stampede3.yaml \
    --years FIRST LAST
```

Look at the `published` and `pending` counts in the output.

Check logs for a running job:

```bash
# List your run directories (most recent last)
ls -lt logs/reforecast/

# Tail the Slurm output of a running task
tail -f logs/reforecast/<run_dir>/slurm/task_0.job_<jobid>.out

# Check per-date timing
tail -5 logs/reforecast/<run_dir>/tasks/task_000/<YYYY-MM-DD>T00/perf.log
```

---

## Step 4: Resubmit Every ~24 Hours

After each wave completes, simply **re-run the same 3 submit commands** for
your year range. The script detects already-published dates and only queues the
remaining ones — no manual bookkeeping needed.

Set a calendar reminder every 24 h to resubmit.
