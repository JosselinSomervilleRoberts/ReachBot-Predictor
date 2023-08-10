# This file permits to launch several times train.py with the same arguments
# Example:
# python tools/train_repeat.py --num_repeats 3 -- --config config_perso.py --amp

import argparse
import os
import subprocess
from toolbox.printing import print_color, warn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="How many times to repeat the training",
    )
    # Enable to have one or several configs
    # To train with one config, use --config config.py
    # To train with several configs, use --config config1.py config2.py
    parser.add_argument(
        "--config",
        nargs="+",
        help="train config file path(s)",
    )
    parser.add_argument(
        "train_args", nargs=argparse.REMAINDER, help="Arguments to pass to train.py"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Remove "--" from args.train_args
    args.train_args = [arg for arg in args.train_args if arg != "--"]

    # Makes sure that all the configs exist
    base_path = os.path.abspath(f"{os.path.dirname(os.path.realpath(__file__))}/..")
    print("\n\n")
    print_color(f"Base path: {base_path}", "blue")
    print_color(f"Training with the following configs:", "blue")
    all_exist = True
    for config in args.config:
        config_path = os.path.join(base_path, config)
        if not os.path.isfile(config_path):
            all_exist = False
            print_color(f"   - {config_path} [DOES NOT EXIST]", "red")
        else:
            print_color(f"   - {config_path} [EXISTS]", "blue")
    if not all_exist:
        warn("Some configs do not exist, exiting")
        exit(1)
    print("\n\n")

    num_configs = len(args.config)
    for i in range(args.num_repeats):
        for j, config in enumerate(args.config):
            print("\n")
            print_color(
                f"Training config {j+1}/{num_configs} - Iter {i+1}/{args.num_repeats}",
                "green",
            )
            print_color(f"   -> Config: {config}", "green")

            # CD to parent directory of this file
            command_to_execute = f"cd {base_path}"
            command_to_execute += " && "
            command_to_execute += (
                f"python tools/train.py {config} {' '.join(args.train_args)}"
            )

            print_color("   -> Executing commands: ", "green")
            for command in command_to_execute.split(" && "):
                print_color(f"        - {command}", "green")
            p = subprocess.Popen(command_to_execute, shell=True, executable="/bin/bash")
            # This makes the wait possible
            p_status = p.wait()


if __name__ == "__main__":
    main()
