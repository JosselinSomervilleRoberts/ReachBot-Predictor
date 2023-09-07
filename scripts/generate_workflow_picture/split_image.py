# Reads an image given as an argument and splits it into N x N tiles.
# Display the tiles in a new image with a grid overlay.

import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="input.png")
    parser.add_argument("--output_path_grid", type=str, default="grid.png")
    parser.add_argument("--output_path_list", type=str, default="list.png")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--paddding", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=256)

    args = parser.parse_args()
    return args


def gen_image(args):
    img = cv2.imread(args.input_path)
    img = np.ones((1024, 1024, 3), dtype=np.uint8) * 127
    size = args.n * args.block_size + (args.n - 1) * args.paddding
    width = img.shape[1]
    height = img.shape[0]
    output_grid = np.ones((size, size, 3), dtype=np.uint8) * 255
    output_list = (
        np.ones(
            (
                (args.n + 1) * args.block_size + args.n * args.paddding,
                args.block_size,
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(args.n):
        x = i * (args.block_size + args.paddding)
        for j in range(args.n):
            y = j * (args.block_size + args.paddding)
            input_block = img[
                i * height // args.n : (i + 1) * height // args.n,
                j * width // args.n : (j + 1) * width // args.n,
                :,
            ]
            # Resize the block to block_size x block_size
            input_block = cv2.resize(
                input_block,
                (args.block_size, args.block_size),
                interpolation=cv2.INTER_LINEAR,
            )
            output_grid[
                x : x + args.block_size, y : y + args.block_size, :
            ] = input_block
            if i == 0 and j != args.n - 1:
                output_list[
                    y : y + args.block_size,
                    :,
                    :,
                ] = input_block
            elif i == args.n - 1 and j == args.n - 1:
                output_list[
                    x + args.block_size + args.paddding :,
                    :,
                ] = input_block

    # Draw 3 circles vertically to represent the ... (etc.)
    height_start = (args.n - 1) * args.block_size + (args.n - 2) * args.paddding
    height_range = 2 * args.paddding + args.block_size
    for i in range(3):
        height_circle = height_start + (i + 1) * height_range / 4
        cv2.circle(
            output_list,
            (args.block_size // 2, int(height_circle)),
            args.paddding // 2,
            (0, 0, 0),
            -1,
        )

    cv2.imwrite(args.output_path_grid, output_grid)
    cv2.imwrite(args.output_path_list, output_list)


if __name__ == "__main__":
    args = parse_args()
    gen_image(args)
