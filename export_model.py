import argparse
import os

import coremltools
import torch
import torch.nn as nn


from timm import create_model
import models


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        else:
            replace_batchnorm(child)


def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--variant", type=str, required=True, help="Provide ffnet model variant name."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Provide location of trained checkpoint.",
    )
    parser.add_argument(
        "--res_h",
        type=int,
        default=256,
        help="Provide resolution of input (height)",
    )
    parser.add_argument(
        "--res_w",
        type=int,
        default=256,
        help="Provide resolution of input (width)",
    )
    return parser


def export(variant: str, output_dir: str, checkpoint: str = None, res_h: int = 256, res_w: int = 256) -> None:
    """Method exports coreml package for mobile inference.

    Args:
        variant: FFNet model variant.
        output_dir: Path to save exported model.
        checkpoint: Path to trained checkpoint. Default: ``None``
    """
    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Random input tensor for tracing purposes.
    inputs = torch.rand(1, 3, res_h, res_w)
    inputs_tensor = [
        coremltools.TensorType(
            name="images",
            shape=inputs.shape,
        )
    ]

    # Instantiate model variant.
    model = create_model(variant)
    if checkpoint is not None:
        print(f"Load checkpoint {checkpoint}")
        chkpt = torch.load(checkpoint)
        model.load_state_dict(chkpt["state_dict"])
    replace_batchnorm(model)
    print(f"Export and Convert Model: {variant}")

    model.eval()

    # Trace and export.
    traced_model = torch.jit.trace(model, torch.Tensor(inputs))
    output_path = os.path.join(output_dir, variant)
    output_path = output_path +  f"_{res_h}" + f"_{res_w}"
    pt_name = output_path + ".pt"
    traced_model.save(pt_name)
    ml_model = coremltools.convert(
        model=pt_name,
        outputs=None,
        inputs=inputs_tensor,
        convert_to="mlprogram",
        debug=False,
    )
    ml_model.save(output_path + ".mlpackage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to export coreml package file")
    parser = parse_args(parser)
    args = parser.parse_args()

    export(args.variant, args.output_dir, args.checkpoint, args.res_h, args.res_w)
