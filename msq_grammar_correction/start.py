import argparse
from pathlib import Path
import sys

import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=str(Path(__file__).parents[1] / "config" / "config.yaml"),
    help="Path to configuration file",
)
args, unknown = parser.parse_known_args(sys.argv[1:])

with open(args.config, "r") as fp:
    config = yaml.load(fp)