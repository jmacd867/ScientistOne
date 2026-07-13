import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .audit.run import run_audit
from .config import load_config
from .llm import LLMClient
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

    audit_p = sub.add_parser("audit", help="run the CoE integrity audit on a run")
    audit_p.add_argument("run_dir", type=Path)
    audit_p.add_argument("--config", type=Path, default=Path("config.yaml"))

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
    elif args.command == "audit":
        config = load_config(args.config)
        llm = LLMClient(config, args.run_dir)
        report = run_audit(llm, config, args.run_dir)
        for check in report.checks:
            mark = {True: "PASS", False: "FAIL", None: "N/A "}[check.passed]
            print(f"[{mark}] {check.name}: {check.detail}")


if __name__ == "__main__":
    main()
