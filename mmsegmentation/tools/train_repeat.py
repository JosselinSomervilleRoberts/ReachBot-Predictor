# This file permits to launch several times train.py with the same arguments
# Example:
# python tools/train_repeat.py --num_repeats 3 -- --config config_perso.py --amp

import argparse
import os
import subprocess
from toolbox.printing import print_color


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="How many times to repeat the training",
    )
    parser.add_argument(
        "train_args", nargs=argparse.REMAINDER, help="Arguments to pass to train.py"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for i in range(args.num_repeats):
        print_color(f"Training {i+1}/{args.num_repeats}", "green")

        # CD to parent directory of this file
        command_to_execute = (
            f"cd {os.path.abspath(f'{os.path.dirname(os.path.realpath(__file__))}/..')}"
        )
        command_to_execute += " && "
        command_to_execute += f"python tools/train.py {' '.join(args.train_args)}"

        print("  - Executing command: ", command_to_execute)
        p = subprocess.Popen(command_to_execute, shell=True, executable="/bin/bash")
        # This makes the wait possible
        p_status = p.wait()


if __name__ == "__main__":
    main()
