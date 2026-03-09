"""
Main entry point for the foil board optimizer.

Usage:
    python -m foilopt single [--config ...]    # Single optimization run
    python -m foilopt research [--max-exp ...]  # Autonomous research loop
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Foil Board Internal Structure Optimizer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single optimization run
    single_parser = subparsers.add_parser("single", help="Run single optimization")
    single_parser.add_argument("--config", default="configs/default.yaml")
    single_parser.add_argument("--output", default="results/single_run")
    single_parser.add_argument("--no-stl", action="store_true")
    single_parser.add_argument("--no-plot", action="store_true")

    # Autonomous research loop
    research_parser = subparsers.add_parser("research", help="Run auto-research loop")
    research_parser.add_argument("--max-experiments", type=int, default=30)
    research_parser.add_argument("--per-generation", type=int, default=4)
    research_parser.add_argument("--output", default="results")

    args = parser.parse_args()

    if args.command == "single":
        # Delegate to run_single
        sys.argv = ["foilopt.run_single"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.output:
            sys.argv.extend(["--output", args.output])
        if args.no_stl:
            sys.argv.append("--no-stl")
        if args.no_plot:
            sys.argv.append("--no-plot")
        from .run_single import main as run_single
        run_single()

    elif args.command == "research":
        from .harness.auto_researcher import AutoResearcher
        researcher = AutoResearcher(output_dir=args.output)
        researcher.run(
            max_experiments=args.max_experiments,
            experiments_per_gen=args.per_generation,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
