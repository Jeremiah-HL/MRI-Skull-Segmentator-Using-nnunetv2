"""Segment the skull on a head-and-neck MR volume with the nnU-Net v2 model shipped in this repository.

usage:
    python predict_skull.py <mr.nii.gz> <skull_mask.nii.gz> [--device cuda|cpu]

The script wraps `nnUNetv2_predict` (Dataset501_SkullSeg, 3d_fullres, fold 4, checkpoint_best.pth). The mask is
written on the grid of the input MR (label 1 = skull).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk

RESULTS = Path(__file__).resolve().parent / "nnUNet_results"
MODEL = RESULTS / "Dataset501_SkullSeg" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_4" / "checkpoint_best.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mr", type=Path, help="input MR volume (any format SimpleITK reads)")
    parser.add_argument("out", type=Path, help="output skull mask (.nii.gz)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        sys.exit(f"Model weights not found at {MODEL}. They are stored with Git LFS: run `git lfs pull` in the repository.")
    predictor = shutil.which("nnUNetv2_predict")
    if predictor is None:
        sys.exit("nnUNetv2_predict not found. Install nnU-Net v2: pip install nnunetv2")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in").mkdir()
        # nnU-Net expects <case>_0000.nii.gz for the single MR channel
        sitk.WriteImage(sitk.Cast(sitk.ReadImage(str(args.mr)), sitk.sitkFloat32), str(tmp / "in" / "case_0000.nii.gz"))
        env = os.environ.copy()
        env["nnUNet_results"] = str(RESULTS)
        env.setdefault("nnUNet_raw", str(tmp / "raw"))                    # not used for prediction; set to avoid warnings
        env.setdefault("nnUNet_preprocessed", str(tmp / "preprocessed"))
        subprocess.run(
            [predictor, "-i", str(tmp / "in"), "-o", str(tmp / "out"), "-d", "501", "-c", "3d_fullres", "-f", "4",
             "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "-chk", "checkpoint_best.pth", "-device", args.device],
            env=env, check=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp / "out" / "case.nii.gz", args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
