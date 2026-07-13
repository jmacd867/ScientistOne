import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline

STAGES = ("investigator", "discovery", "writer")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scientist-one")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full pipeline on a task")
    run_p.add_argument("--task", required=True, type=Path)
    run_p.add_argument("--config", type=Path, default=Path("config.yaml"))
    run_p.add_argument("--run-dir", type=Path, default=None,
                       help="existing run dir resumes; default runs/<timestamp>")

    status_p = sub.add_parser("status", help="show stage completion for a run")
    status_p.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        run_dir = args.run_dir or Path("runs") / datetime.now(
            timezone.utc).strftime("%Y%m%d-%H%M%S")
        manifest = run_pipeline(load_config(args.config), args.task, run_dir)
        print(json.dumps(manifest, indent=2))
        print(f"\nrun dir: {run_dir}")
    elif args.command == "status":
        for stage in STAGES:
            done = (args.run_dir / f"{stage}.json").exists()
            print(f"{stage:14} {'done' if done else 'pending'}")
        manifest = args.run_dir / "manifest.json"
        if manifest.exists():
            print(json.loads(manifest.read_text())["status"])


if __name__ == "__main__":
    main()
