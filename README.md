# PoM2-DIB: Probabilistic Multi-Modal Multi-Task Distributed Information Bottleneck

PyTorch implementation of **"Multi-Modal Multi-Task Semantic Communication: A Distributed Information Bottleneck Perspective"**.

*Preprint:* [arXiv:2510.04000](https://arxiv.org/abs/2510.04000)  
*Venue:* Under review at IEEE Transactions on Communications (preliminary version presented at IEEE ICC 2025)

## Overview

This repository contains the code implementation for PoM2-DIB, a framework that extends distributed information bottleneck (DIB) theory to multi-modal multi-task semantic communication systems. Our approach addresses the challenge of efficient communication across distributed multi-modal sources under physical resource constraints manifested as the limited number of communication links.

### Key Features

- *Probabilistic Modality Selection*: Introduces modality selection as a new degree of freedom in the rate-relevance tradeoff
- *Distributed Coordination*: Leverages common randomness for coordinated yet decentralized decision-making
- *End-to-End Optimization*: Jointly optimizes semantic codecs and selection policies

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.2.2+
- CUDA 13.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SamuChamp/pom2dib.git
cd pom2dib
```

2. Install required an external package:

*NPEET* (for entropy estimation):
```bash
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install -e .
cd ..
```

## Dataset

### HandWritten (AV-MNIST)

This dataset combines MNIST handwritten digits with FSDD spoken digits.

1. **MNIST**: Download automatically via torchvision
2. **FSDD** (Free Spoken Digit Dataset):
```bash
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
```

The code will automatically align MNIST images with FSDD audio samples and apply augmentation.

### MM-Fi

Multi-modal human activity dataset with RGB-D, LiDAR, mmWave, and WiFi-CSI modalities.

1. Download from: https://github.com/ybhbingo/MMFi_dataset
2. Follow the MM-Fi toolbox instructions for data preprocessing
3. Place processed data in `data/mmfi/`

**Dataset Structure:**
```
data/
|- FSDD/
    |- recordings/
|- mnist/
    |- (auto-downloaded)
|- mmfi/
    |- (processed MM-Fi data)
```

## Usage

### Quick Start

Train PoM2-DIB on HandWritten dataset with default settings:
```bash
python main.py --dataset handwritten
```

Train PoM2-DIB on MM-Fi dataset:
```bash
python main.py --dataset mmfi
```

### Training Different Methods

*PoM2-DIB* (Proposed method with optimizable selection):
```bash
python main.py --dataset handwritten --no_det True --sampling opt
```

*RS-DIB* (Random selection baseline):
```bash
python main.py --dataset handwritten --no_det True --sampling rand
```

*TADIB* (Full participation baseline):
```bash
python main.py --dataset handwritten --no_det True --sampling full
```

*DLSC* (Deep learning-based SemCom without rate control):
```bash
python main.py --dataset handwritten --no_det False --sampling full
```

### Advanced Usage

**Explore rate-relevance tradeoff** by varying beta:
```bash
for beta in 1e-4 1e-3 1e-2 1e-1; do
    python main.py --dataset handwritten --beta $beta
done
```

**Study learning rate ratio effects**:
```bash
python main.py --dataset handwritten --lr_code 1e-4 --lr_sel 0.5e-4  # ratio 1:2 (default)
```

**Enable sparse selection without hard constraints** (three-way tradeoff):
```bash
python main.py --dataset handwritten --sparse True --gamma 1e-3
```

## Project Structure

```
pom2dib/
|- main.py               # Main training script
|- config.py             # Configuration and argument parser
|- model/
   |- __init__.py
   |- data.py            # Dataset loaders (HandWritten, MMFi)
   |- environ.py         # Environment functions (selection, MI estimation)
   |- functionals.py     # Loss functions and metrics
   |- modules.py         # Transmitter and Receiver modules
   |- nets.py            # Neural network architectures
|- utils/
   |- __init__.py
   |- utils.py           # Utility functions
|- data/                 # Dataset directory
   |- FSDD/
   |- mnist/
   |- mmfi/
|- logs/                 # Training logs
|- README.md
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{zhou2025multimodal,
  title={Multi-Modal Multi-Task Semantic Communication: A Distributed Information Bottleneck Perspective},
  author={Zhou, Yujie and Liao, Yiwei and Peng, Cheng and Wang, Rulong and Xiao, Yong and Li, Yingyu and Shi, Guangming},
  journal={arXiv preprint arXiv:2510.04000},
  year={2025}
}
```

## Acknowledgements

We thank the authors of [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset), [MM-Fi](https://github.com/ybhbingo/MMFi_dataset), and [NPEET](https://github.com/gregversteeg/NPEET) for making their datasets and tools publicly available.
