"""
Command-line interface for pdf-to-wiki.

Currently exposes the minimum needed to run the pipeline:
  - positional input (PDF path)
  - --seeds flag (domain seed file)

Additional flags (--output, --provider, --model, --workers, etc.)
will be added in feature/v2-cli per the Step 3 roadmap.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-to-wiki",
        description="Convert a PDF into interlinked wiki concept pages.",
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        metavar="PDF",
        help="Path to the PDF file to process.",
    )

    parser.add_argument(
        "--seeds",
        metavar="FILE",
        default=None,
        help=(
            "Path to a JSON file containing seed concept names (array of strings). "
            "Skips auto-seed generation (Pass 1.5). "
            "Example: seeds/accounting.json"
        ),
    )

    parser.add_argument(
        "--output",
        metavar="DIR",
        default="./vault",
        help=(
            "Directory to write concept pages into. "
            "Each page is saved as its own .md file. "
            "Created if it does not exist. Default: ./vault"
        ),
    )

    return parser


def apply_args_to_env(args: argparse.Namespace) -> None:
    """
    Write explicitly-set CLI args into os.environ so the existing
    env-reading code in factory.py picks them up without modification.

    'input' and 'seeds' are handled directly in main.py — no env mapping needed.
    This function is a placeholder for the flags added in feature/v2-cli.
    """
