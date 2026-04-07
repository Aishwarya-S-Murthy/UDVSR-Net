
<h1 align="left">An Adaptive Spectral Upsampling Framework for Underwater Degraded Video Super-Resolution Benchmark</h1>

<!-- <p align="center">
  <a href="https://arxiv.org/abs/YOUR_PAPER_ID"><img src="https://img.shields.io/badge/arXiv-Paper-red"/></a>
  <a href="https://github.com/Aishwarya-S-Murthy/UDVSR-Net"><img src="https://img.shields.io/github/stars/Aishwarya-S-Murthy/UDVSR-Net?style=social"/></a>
</p> -->


<!-- <p align="center">
  <img src="assets/architecture.png" width="800"/>
</p>

> *Figure: Overall architecture of UDVSRNet.* -->

## Installation

Please refer [INSTALL.md](INSTALL.md) for the installation of dependencies required to run UDVSRNet.

## Datasets
Download the dataset from the link below and place it in the `Datasets/` folder:
> **[Download UDVSR Dataset](https://drive.google.com/drive/folders/1R4kZX1Y9XmFqEpeoOWVuYt7u0QCRDjwl?usp=drive_link)**

- `input/` contains the **low-resolution underwater video frames**
- `GT/GT_x2`, `GT/GT_x3`, `GT/GT_x4` contain the **ground truth high-resolution frames** at ×2, ×3, and ×4 scale factors respectively
- Each video is organized as a **separate folder** (e.g., `video01/`, `video02/`, ...) containing individual frames named as `frame000.png`, `frame001.png`, etc.


After downloading, organize the dataset as follows:
```
UDVSR-Net/
├── basicsr/
├── Underwater Video SR/
    └── Datasets/
        ├── train/
        │   ├── input/
        │   │   ├── video01/
        │   │   │   ├── frame000.png
        │   │   │   ├── frame001.png
        │   │   │   └── ...
        │   │   └── ...
        │   └── GT/
        │       ├── GT_x2/
        │       │   ├── video01/
        │       │   │   ├── frame000.png
        │       │   │   ├── frame001.png
        │       │   │   └── ...
        │       │   └── ...
        │       ├── GT_x3/
        │       │   ├── video01/
        │       │   │   ├── frame000.png
        │       │   │   ├── frame001.png
        │       │   │   └── ...
        │       │   └── ...
        │       └── GT_x4/
        │           ├── video01/
        │           │   ├── frame000.png
        │           │   ├── frame001.png
        │           │   └── ...
        │           └── ...
        └── test/
            ├── input/
            │   ├── video01/
            │   │   ├── frame000.png
            │   │   ├── frame001.png
            │   │   └── ...
            │   └── ...
            └── GT/
                ├── GT_x2/
                │   ├── video01/
                │   │   ├── frame000.png
                │   │   ├── frame001.png
                │   │   └── ...
                │   └── ...
                ├── GT_x3/
                │   ├── video01/
                │   │   ├── frame000.png
                │   │   ├── frame001.png
                │   │   └── ...
                │   └── ...
                └── GT_x4/
                    ├── video01/
                    │   ├── frame000.png
                    │   ├── frame001.png
                    │   └── ...
                    └── ...
```
## Training

To train UDVSRNet from scratch, run the following command:
```bash
python -m basicsr.train -opt "Underwater Video SR/Options/Underwater Video SR.yml"
```

---

## Testing

To test the pre-trained UDVSRNet model, run the following command:
```bash
python -m basicsr.test -opt "Underwater Video SR/Options/Underwater Video SR_test.yml"
```
## Pre-trained Models
We provide pre-trained model checkpoints trained on our proposed UDVSR Dataset.
| BasicVSR | EfficientU | URSCST | Madnet | RDG-s | UDVSRNet (Ours) |
|----------|------------|--------|--------|-------|-----------------|
|          |            |        |        |       | [x2](https://drive.google.com/drive/folders/1eAvv7xm0v1LKuQ0_uqEiMlJSqGfDWuoZ?usp=drive_link)               |
|          |            |        |        |       |  [x3](https://drive.google.com/drive/folders/1eAvv7xm0v1LKuQ0_uqEiMlJSqGfDWuoZ?usp=drive_link)               |
|          |            |        |        |       |  [x4](https://drive.google.com/drive/folders/1eAvv7xm0v1LKuQ0_uqEiMlJSqGfDWuoZ?usp=drive_link)               |

## Contact
For any questions about this work, please contact a.murthy@iitg.ac.in.
## Acknowledgment: 
This code is based on the [Restormer] (https://github.com/swz30/Restormer).

