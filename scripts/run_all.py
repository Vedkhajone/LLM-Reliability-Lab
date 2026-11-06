"""Run all six experiments (same options as run_experiment.py, minus --experiment)."""
import sys

import run_experiment

if __name__ == "__main__":
    sys.exit(run_experiment.main(all_experiments=True))
