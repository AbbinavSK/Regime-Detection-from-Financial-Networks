import argparse
import subprocess
import sys
import time
from pathlib import Path

from regime_networks.data import required_data_files, missing_data_message

REPO_ROOT = Path(__file__).resolve().parent
DATA_STEPS = ("verify", "networks", "sweeps", "portfolio")
STEPS = [
    ("download", "scripts/download_data.py", []),
    ("manifest", "scripts/data_manifest.py", []),
    ("verify", "scripts/data_manifest.py", ["--verify"]),
    ("networks", "scripts/data_processing.py", []),
    ("sweeps", "scripts/run_sweeps.py", ["--descriptor", "all"]),
    ("geometry", "scripts/embedding_geometry_sweep.py", []),
    ("topology", "scripts/network_topology.py", []),
    ("model_selection", "scripts/model_selection.py", []),
    ("seed_robustness", "scripts/seed_robustness.py", []),
    ("portfolio", "scripts/portfolio_analysis.py", []),
    ("regime_figures", "figures/regime_figures.py", []),
    ("geometry_figures", "figures/embedding_geometry_figures.py", []),
    ("topology_figures", "figures/network_topology_figures.py", []),
    ("model_selection_figures", "figures/model_selection_figures.py", []),
    ("portfolio_figures", "figures/portfolio_figures.py", []),
    ("tables", "tables/paper_tables.py", []),
]
STEP_NAMES = [name for name, _, _ in STEPS]


# Names of the steps to run: by default the published data in data/ are verified rather than re-downloaded (a fresh download re-adjusts prices).
def select_steps(download=False, start=None, only=None):
    if only:
        return list(only)

    names = [name for name in STEP_NAMES if name not in (("verify",) if download else ("download", "manifest"))]

    if start and start not in names:
        raise SystemExit(f"--from {start} is not among the selected steps: {names}")

    return names[names.index(start):] if start else names


# Run the selected steps in order as subprocesses, stopping at the first failure since every later step depends on earlier outputs, and print a timing summary; a run that reads data/ without downloading it first stops before any step if a data file is missing.
def main(names):
    missing = [path for path in required_data_files() if not path.exists()] if "download" not in names and set(DATA_STEPS) & set(names) else []

    if missing:
        print(missing_data_message(missing))
        return False

    steps = {name: (script, args) for name, script, args in STEPS}
    completed = []

    for name in names:
        script, args = steps[name]
        print(f"\n===== {name}: {script} {' '.join(args)} =====", flush=True)
        start = time.time()
        ok = subprocess.run([sys.executable, str(REPO_ROOT / script), *args], cwd=REPO_ROOT).returncode == 0
        completed.append((name, ok, time.time() - start))

        if not ok:
            break

    print("\n===== summary =====")

    for name, ok, elapsed in completed:
        print(f"{name:25s} {'OK' if ok else 'FAILED':6s} {elapsed / 60:7.1f} min")

    if not all(ok for _, ok, _ in completed):
        print(f"stopped at {completed[-1][0]}; fix it and resume with --from {completed[-1][0]}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce every output, figure and table of the paper from data/ (or from a fresh download) in dependency order.")
    parser.add_argument("--download", action="store_true", help="Re-download and re-screen all prices and write a new data manifest, instead of verifying the existing data against it")
    parser.add_argument("--from", dest="start", choices=STEP_NAMES, help="Resume from this step, skipping the earlier ones")
    parser.add_argument("--only", nargs="+", choices=STEP_NAMES, help="Run just these steps, in the order given")
    parser.add_argument("--list", action="store_true", help="Print the steps that would run and exit")
    args = parser.parse_args()
    names = select_steps(args.download, args.start, args.only)

    if args.list:
        print("\n".join(f"{name:25s} {dict((n, s) for n, s, _ in STEPS)[name]}" for name in names))
        sys.exit(0)

    sys.exit(0 if main(names) else 1)
