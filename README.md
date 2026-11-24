# MRI-Skull-Segmentator-Using-nnunetv2

An nnU-Net v2 model that segments the skull on head-and-neck MR images, together with the way it was trained and a
script to run it.

Bone has no direct intensity signature in MR, so no CT-style threshold works. This model gives a coarse MR skull mask.
It was built as the MR side of a bone-mask-to-bone-mask affine initialisation for MR–CT registration (the CT side comes
from TotalSegmentator), where a coarse mask is sufficient. It is not meant as a precise skull segmentation.

## Contents

```text
predict_skull.py                 run the model on one MR volume
requirements.txt
nnUNet_results/
└── Dataset501_SkullSeg/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        ├── dataset.json
        ├── plans.json
        ├── dataset_fingerprint.json
        └── fold_4/
            ├── checkpoint_best.pth      model weights (Git LFS, 247 MB)
            └── progress.png             training curves
```

The weights are stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning, or run `git lfs pull`
afterwards; otherwise `checkpoint_best.pth` is only a small pointer file.

## Usage

```bash
pip install -r requirements.txt        # install PyTorch for your CUDA version first, see pytorch.org
python predict_skull.py path/to/mr.nii.gz path/to/skull_mask.nii.gz
```

Add `--device cpu` to run without a GPU. The output is a label map on the grid of the input MR (1 = skull,
0 = background).

The script is a thin wrapper around the standard nnU-Net command. The equivalent call is:

```bash
export nnUNet_results=/path/to/this/repo/nnUNet_results
# the input folder must contain <case>_0000.nii.gz
nnUNetv2_predict -i input_dir -o output_dir -d 501 -c 3d_fullres -f 4 \
    -tr nnUNetTrainer -p nnUNetPlans -chk checkpoint_best.pth
```

## Method

The training set was built from the public [HaN-Seg](https://doi.org/10.1002/mp.16197) head-and-neck CT and MR
images. There is no manual skull annotation on MR, so the labels are pseudo-labels taken from the CT:

1. **Affine pre-registration.** For each HaN-Seg case, the CT and the MR are aligned with an affine registration in
   [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (`reg_aladin`).
2. **Label transfer.** The skull label of the CT is used as the label of the aligned MR.
3. **Dataset and training.** The aligned MR volumes and their labels are written as an nnU-Net v2 dataset and the
   default nnU-Net pipeline is trained on it.

The resulting dataset has 82 MR volumes with one channel (`MRI`) and one foreground label (`skull`), at 1 mm isotropic
spacing (median size 128 x 224 x 224 voxels). The dataset itself is not included in this repository. nnU-Net dataset layout:

```text
nnUNet_raw/Dataset501_SkullSeg/
├── dataset.json
├── imagesTr/skullseg_0001_0000.nii.gz ...
└── labelsTr/skullseg_0001.nii.gz ...
```

**Training.** Default nnU-Net v2 pipeline, no custom trainer or plans:

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 3d_fullres 4
```

| Setting | Value |
|---|---|
| Configuration | `3d_fullres`, `nnUNetTrainer`, `nnUNetPlans` |
| Network | PlainConvUNet, 6 stages, features 32–64–128–256–320–320 |
| Patch size / batch size | 96 × 160 × 160 / 2 |
| Normalisation | z-score |
| Epochs | 1000, initial learning rate 0.01 |
| Split | fold 4 of nnU-Net's 5-fold split: 66 training, 16 validation cases |
| Hardware | one NVIDIA RTX 3090 |

Only fold 4 was trained. `checkpoint_best.pth` is the checkpoint with the best moving-average validation Dice
(0.806, epoch 954).

## Validation

nnU-Net's final validation on the 16 held-out cases of fold 4, measured against the pseudo-labels:

| Dice | Value |
|---|---|
| Mean | 0.778 |
| Median | 0.866 |
| Range | 0.033 – 0.914 |

Most cases reach a Dice of about 0.8 to 0.9: 12 of the 16 cases are above 0.85. Three cases are between 0.58 and 0.69 and one case fails (0.033). Because the
references are pseudo-labels transferred from CT after a coarse alignment, these numbers also contain label error and
should not be read as accuracy against a manual ground truth.

## Limitations

- Trained on HaN-Seg head-and-neck MR only; behaviour on other sequences, scanners or body regions is untested.
- Trained from pseudo-labels, with one fold only.
- Intended for coarse bone masks (e.g. registration initialisation), not for measurement or clinical use.
- The training images are not redistributed here; HaN-Seg is available from its authors under its own licence.

## Acknowledgement

The training data are derived from HaN-Seg:

> Podobnik, G., Strojan, P., Peterlin, P., Ibragimov, B., & Vrtovec, T. (2023). HaN-Seg: The head and neck
> organ-at-risk CT and MR segmentation dataset. *Medical Physics*, 50(3), 1917-1927.

The model is trained with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet):

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
