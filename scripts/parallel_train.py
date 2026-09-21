import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import run as run_plan
import train as train_script


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MFC training grid with parallel workers.")
    parser.add_argument(
        "--env",
        choices=run_plan.PRIMARY_ENVS + ["all"],
        required=True,
    )
    parser.add_argument("--seeds", type=run_plan.parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--budget-mode", choices=["fair", "manual"], default="fair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=None)
    parser.add_argument("--simplex-resolution", type=int, default=None)
    parser.add_argument("--q-learning-lr-power", type=float, default=None)
    parser.add_argument("--q-learning-sampling", choices=["sweep", "iid"], default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--jobs-file", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--auto-workers", action="store_true")
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--target-cpu", type=float, default=90.0)
    parser.add_argument("--target-gpu", type=float, default=None)
    parser.add_argument("--min-free-memory-gb", type=float, default=6.0)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_against_root(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return run_plan.ROOT / path


def normalize_paths(args):
    args.results_root = str(resolve_against_root(args.results_root))
    if args.jobs_file is not None:
        args.jobs_file = resolve_against_root(args.jobs_file)
    if args.logs_root is not None:
        args.logs_root = resolve_against_root(args.logs_root)


def train_args_for(job_spec, seed, args):
    return SimpleNamespace(
        env=job_spec["env"],
        algorithm=job_spec["algorithm"],
        perturbation=job_spec["perturbation"],
        eta=args.eta if args.eta is not None else job_spec.get("eta"),
        horizon=job_spec["horizon"],
        flow=job_spec["flow"],
        seed=seed,
        results_root=args.results_root,
        simplex_resolution=args.simplex_resolution or 30,
        # output_directory separates the mixture sizes, so resume and dedup have
        # to see the same value the launcher puts on the command line.
        n_components=job_spec.get("n_components") or args.n_components,
    )


def log_path_for(output_dir, results_root, logs_root):
    try:
        relative = output_dir.relative_to(results_root)
    except ValueError:
        relative = Path(*output_dir.parts[1:]) if output_dir.is_absolute() else output_dir
    return logs_root / relative.parent / f"{relative.name}.log"


def build_records(args):
    selected_envs = run_plan.PRIMARY_ENVS if args.env == "all" else [args.env]
    results_root = Path(args.results_root)
    logs_root = args.logs_root or results_root / "logs"
    records = []
    seen_output_dirs = set()

    for env in selected_envs:
        for job_spec in run_plan.experiment_plan(env):
            for seed in args.seeds:
                train_args = train_args_for(job_spec, seed, args)
                output_dir = train_script.output_directory(train_args)
                if output_dir in seen_output_dirs:
                    continue
                seen_output_dirs.add(output_dir)
                records.append(
                    {
                        "command": run_plan.command_for(job_spec, seed, args),
                        "environment": worker_environment(args.threads_per_worker),
                        "output_dir": output_dir,
                        "summary_path": output_dir / "summary.json",
                        "log_path": log_path_for(output_dir, results_root, logs_root),
                    }
                )

    return records


def write_jobs_file(records, jobs_file, resume):
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with jobs_file.open("w", encoding="utf-8") as file:
        for record in records:
            completed = record["summary_path"].exists()
            payload = {
                "status": "skipped" if resume and completed else "pending",
                "command": [str(value) for value in record["command"]],
                "output_dir": str(record["output_dir"]),
                "summary_path": str(record["summary_path"]),
                "log_path": str(record["log_path"]),
            }
            file.write(json.dumps(payload) + "\n")


def worker_environment(threads):
    """Thread budget of one worker process.

    The estimator works on small tensors, so intra-op parallelism buys nothing:
    one training run measures the same at one thread as at eight. The throughput
    of a grid is therefore the number of concurrent runs, and leaving every
    worker free to grab every core only produces oversubscription.
    """
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(threads)
    return environment


def run_record(record, index, total, print_lock):
    command = [str(value) for value in record["command"]]
    log_path = record["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    with print_lock:
        print(f"[{index}/{total}] start {record['output_dir']} -> {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + json.dumps(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=run_plan.ROOT,
            env=record["environment"],
        )
        elapsed = time.perf_counter() - started_at
        log.write(f"\nexit_code={result.returncode} elapsed_seconds={elapsed:.3f}\n")

    if result.returncode != 0:
        raise RuntimeError(f"Job failed with exit code {result.returncode}: {record['output_dir']} ({log_path})")

    with print_lock:
        print(f"[{index}/{total}] done  {record['output_dir']} ({elapsed:.1f}s)", flush=True)


def run_pending(pending, workers):
    print_lock = threading.Lock()
    indexed = [(index, record) for index, record in enumerate(pending, start=1)]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_record, record, index, len(pending), print_lock): record
            for index, record in indexed
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((record, error))
                with print_lock:
                    print(f"FAILED {record['output_dir']} ({record['log_path']}): {error}", flush=True)

    if failures:
        print("\nFailed jobs:", flush=True)
        for record, error in failures:
            print(f"- {record['output_dir']} ({record['log_path']}): {error}", flush=True)
        raise RuntimeError(f"{len(failures)} job(s) failed; see per-job logs above.")


def read_cpu_times():
    with Path("/proc/stat").open("r", encoding="utf-8") as file:
        fields = file.readline().split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


def cpu_percent(previous, current):
    previous_idle, previous_total = previous
    current_idle, current_total = current
    total_delta = current_total - previous_total
    if total_delta <= 0:
        return 0.0
    idle_delta = current_idle - previous_idle
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def available_memory_gb():
    with Path("/proc/meminfo").open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    return float("inf")


def gpu_percent():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return max(values) if values else None


def print_failures(failures):
    if not failures:
        return
    print("\nFailed jobs:", flush=True)
    for record, error in failures:
        print(f"- {record['output_dir']} ({record['log_path']}): {error}", flush=True)


def run_pending_auto_workers(pending, args):
    print_lock = threading.Lock()
    queue = deque((index, record) for index, record in enumerate(pending, start=1))
    failures = []
    active_limit = min(args.min_workers, args.workers, len(pending))
    previous_cpu_times = read_cpu_times()

    def submit_until_limit(executor, futures):
        while queue and len(futures) < active_limit:
            index, record = queue.popleft()
            future = executor.submit(run_record, record, index, len(pending), print_lock)
            futures[future] = record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        submit_until_limit(executor, futures)
        while futures:
            done, _ = wait(futures, timeout=args.sample_interval, return_when=FIRST_COMPLETED)
            for future in done:
                record = futures.pop(future)
                try:
                    future.result()
                except Exception as error:
                    failures.append((record, error))
                    with print_lock:
                        print(f"FAILED {record['output_dir']} ({record['log_path']}): {error}", flush=True)

            current_cpu_times = read_cpu_times()
            current_cpu = cpu_percent(previous_cpu_times, current_cpu_times)
            previous_cpu_times = current_cpu_times
            current_memory = available_memory_gb()
            current_gpu = gpu_percent() if args.target_gpu is not None else None

            memory_ok = current_memory >= args.min_free_memory_gb
            cpu_low = current_cpu < args.target_cpu
            gpu_low = args.target_gpu is None or current_gpu is None or current_gpu < args.target_gpu
            can_grow = queue and active_limit < args.workers and memory_ok and cpu_low and gpu_low
            should_shrink = (
                active_limit > args.min_workers
                and (not memory_ok or current_cpu > args.target_cpu + 5.0)
            )

            if can_grow:
                active_limit += 1
            elif should_shrink:
                active_limit -= 1

            with print_lock:
                gpu_text = "n/a" if current_gpu is None else f"{current_gpu:.1f}%"
                print(
                    "auto-workers "
                    f"active={len(futures)} limit={active_limit}/{args.workers} "
                    f"queued={len(queue)} cpu={current_cpu:.1f}% gpu={gpu_text} "
                    f"mem_avail={current_memory:.1f}GB",
                    flush=True,
                )

            submit_until_limit(executor, futures)

    if failures:
        print_failures(failures)
        raise RuntimeError(f"{len(failures)} job(s) failed; see per-job logs above.")


def main():
    args = parse_args()
    normalize_paths(args)
    if args.baseline and args.no_baseline:
        raise ValueError("Use at most one of --baseline and --no-baseline.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.min_workers < 1:
        raise ValueError("--min-workers must be at least 1.")
    if args.min_workers > args.workers:
        raise ValueError("--min-workers must be less than or equal to --workers.")

    records = build_records(args)
    if not records:
        raise RuntimeError("The experiment plan produced zero jobs; check --env and --seeds.")

    resume = not args.no_resume
    jobs_file = args.jobs_file or Path(args.results_root) / "train_jobs.jsonl"
    write_jobs_file(records, jobs_file, resume)

    completed = [record for record in records if record["summary_path"].exists()]
    pending = [record for record in records if not (resume and record["summary_path"].exists())]

    print(f"Prepared {len(records)} jobs.")
    print(f"Jobs file: {jobs_file}")
    print(f"Completed jobs skipped: {len(completed) if resume else 0}")
    print(f"Pending jobs: {len(pending)}")
    print(f"Workers: {args.workers}")
    if args.auto_workers:
        print(
            "Adaptive workers: "
            f"min={args.min_workers}, target_cpu={args.target_cpu:g}%, "
            f"target_gpu={args.target_gpu if args.target_gpu is not None else 'off'}, "
            f"min_free_memory_gb={args.min_free_memory_gb:g}",
            flush=True,
        )

    if args.dry_run:
        return
    if not pending:
        print("Nothing to run; every job already has summary.json.")
        return

    if args.auto_workers:
        run_pending_auto_workers(pending, args)
    else:
        run_pending(pending, args.workers)


if __name__ == "__main__":
    main()
