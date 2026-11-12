"""Command-line access to the shared project tools."""

import argparse
import json
from pathlib import Path

from setcover.bounds import counting_bound, lp_bound
from setcover.generator import generate
from setcover.io import read_instance, write_orlib
from setcover.registry import METHODS
from setcover.runner import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Unweighted minimum set cover tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="List registered algorithms")

    run = commands.add_parser("run", help="Run an algorithm and verify its result")
    run.add_argument("--method", choices=sorted(METHODS), required=True)
    run.add_argument("--instance", type=Path, required=True)
    run.add_argument("--time-limit", type=float, default=10.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", default="", help="Optional JSON result file")

    generator = commands.add_parser("generate", help="Generate an OR-Library instance")
    generator.add_argument("--elements", type=int, default=20)
    generator.add_argument("--subsets", type=int, default=40)
    generator.add_argument("--density", type=float, default=0.2)
    generator.add_argument("--seed", type=int, default=0)
    generator.add_argument("--output", type=Path, required=True)

    bounds = commands.add_parser("bounds", help="Compute counting and LP lower bounds")
    bounds.add_argument("--instance", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "methods":
            print("\n".join(sorted(METHODS)))
        elif args.command == "run":
            record = run_one(args.method, args.instance, args.seed, args.time_limit)
            text = json.dumps(record, indent=2)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n")
            print(text)
            if not record["valid"]:
                raise SystemExit(1)
        elif args.command == "generate":
            inst = generate(args.elements, args.subsets, args.density, args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_orlib(inst, args.output)
            print(f"Instance written to {args.output}")
        elif args.command == "bounds":
            inst = read_instance(args.instance)
            print(json.dumps({
                "instance": inst.name,
                "counting_bound": counting_bound(inst),
                "lp_bound": lp_bound(inst),
            }, indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
