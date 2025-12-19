# 🔬 Efficient Prompt-Free Surgical Tools Segmentation and Classification via Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIMI-orange.svg)](https://ieeexplore.ieee.org)

> **Master's Thesis** - Biomedical Engineering, Politecnico di Milano  
> **Author**: Marco De Zen  
> **Advisors**: Prof. Elena De Momi, Mattia Magro  
> **Status**: Submitted to IEEE Transactions on Medical Imaging

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Citation](#-citation)

---

## 🎯 Overview

This project introduces **CMT-Unet**, a lightweight transformer-based model for **real-time surgical instrument segmentation and classification** in robot-assisted surgery (RAS). Through a novel **multi-stage knowledge distillation framework**, we transfer capabilities from SAM (Segment Anything Model) to a compact architecture suitable for deployment in time-critical surgical environments.

### 🚨 Problem Statement

Existing foundation models like SAM face critical limitations in surgical applications:
- ❌ **Require manual prompts** (points, bounding boxes, text)
- ❌ **Inference time > 1 second** (unsuitable for real-time surgery)
- ❌ **Only binary segmentation** (no instrument classification)
- ❌ **637M parameters** (too large for edge deployment)

### ✅ Our Solution

**CMT-Unet** overcomes these limitations with:
- ✅ **Prompt-free** automatic segmentation
- ✅ **Real-time inference** (<25 ms per frame)
- ✅ **Instance segmentation + classification** (7 instrument types)
- ✅ **70× fewer parameters** than SAM

---

## 🌟 Key Features

### **Performance Highlights**

| Metric | CMT-Unet-Large | CMT-Unet-Small | SAM (ViT-H) |
|--------|----------------|----------------|-------------|
| **mIoU** | 0.8919 ± 0.135 | 0.8703 ± 0.120 | 0.8915 ± 0.090 |
| **Dice** | 0.9355 ± 0.121 | 0.9251 ± 0.106 | 0.9393 ± 0.095 |
| **Latency** | **21.78 ms** (~47 Hz) | **15.48 ms** (~65 Hz) | 1063.5 ms (~1 Hz) |
| **Parameters** | 320M | 78M | 637M |
| **FLOPs** | 42 GFLOPs | 34 GFLOPs | 2980 GFLOPs |

**📊 Key Achievements:**
- 🏆 **~50× faster** than SAM with comparable accuracy
- 🏆 **70× fewer operations** (42 GFLOPs vs 2980 GFLOPs)
- 🏆 **No statistical difference** in IoU/Dice vs SAM (p > 0.05)
- 🏆 **Classification accuracy: 95.05%** on 7 instrument types

---

## 🏗️ Architecture

### **Teacher Model: SAM (ViT-H)**

```
Input (1024×1024) → ViT-H Encoder → Prompt Encoder → Mask Decoder → Binary Mask
                     (637M params)
```

### **Student Model: CMT-Unet**

```
Input (1024×1024) → CMT Encoder → CNN Decoder → Segmentation Mask
                     (78M-320M)     (U-Net-style)
                                          ↓
                               EfficientNet-B0 → Classification Head
                                                  (7 instrument classes)
```

#### **CMT Encoder Components:**
- **Convolutional Stem**: Initial feature extraction
- **4 Stages** of CMT blocks:
  - Local Perception Units (LPU)
  - Depthwise Convolutions
  - Efficient Multi-Head Self-Attention (MHSA)

#### **CNN Decoder:**
- 4 upsampling blocks with skip connections
- Progressive spatial resolution recovery
- Inspired by U-Net architecture

#### **Classification Head:**
- EfficientNet-B0 backbone
- Fully connected layers + SE blocks
- Dropout for regularization

---

## 📊 Results

### **Segmentation Performance (MICCAI Test Set)**

| Model | Time (ms) | IoU | Dice | Sensitivity |
|-------|-----------|-----|------|-------------|
| **SAM (ViT-H)** | 1063.5 ± 22.1 | 0.8915 ± 0.09 | 0.9393 ± 0.095 | 0.9457 ± 0.115 |
| EdgeSAM | 74.42 ± 6.3 | 0.8581 ± 0.107*** | 0.9194 ± 0.090*** | 0.9662 ± 0.059*** |
| MobileSAM | 67.79 ± 6.1 | 0.8628 ± 0.095*** | 0.9230 ± 0.085*** | 0.9439 ± 0.081*** |
| FastSAM | 134.67 ± 5.2 | 0.7909 ± 0.262*** | 0.8484 ± 0.246*** | 0.9035 ± 0.162*** |
| YOLOv8-N | 28.69 ± 17.7 | 0.8469 ± 0.125*** | 0.9106 ± 0.111*** | 0.9348 ± 0.146 |
| **CMT-Unet-Large** | **21.78 ± 2.3** | **0.8919 ± 0.135** | **0.9355 ± 0.121** | **0.9624 ± 0.062** |
| **CMT-Unet-Small** | **15.48 ± 2.0** | **0.8703 ± 0.120** | **0.9251 ± 0.106** | **0.9815 ± 0.062** |

*Statistical significance: *** p < 0.001 vs CMT-Unet-Large*

### **Classification Performance**

| Instrument Class | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| Large needle driver | 0.9392 | 0.9931 | 0.8104 | 0.8932 |
| Forceps | 0.8860 | 0.8091 | 0.9813 | 0.8872 |
| Maryland bipolar forceps | 0.9926 | 0.9701 | 0.9031 | 0.9350 |
| Monopolar curved scissors | 0.9920 | 0.8894 | 1.0000 | 0.9415 |
| Vessel sealer | 0.9425 | 0.9084 | 0.5274 | 0.6671 |
| **Macro Average** | **0.9505 ± 0.043** | **0.914 ± 0.07** | **0.8444 ± 0.184** | **0.8648 ± 0.104** |

**Total inference time (segmentation + classification):** ~50 ms (~20 Hz)

---

## 🚀 Installation

### **Requirements**

```bash
Python >= 3.8
PyTorch >= 2.0.0
CUDA >= 11.7 (for GPU acceleration)
```

### **Setup**

```bash
# Clone repository
git clone https://github.com/mthezn/surgical-tool-segmentation.git
cd surgical-tool-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
python scripts/download_weights.py --model cmt-unet-large
```

### **Dependencies**

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
albumentations>=1.3.0
pillow>=9.5.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
wandb>=0.15.0  # For experiment tracking
pydicom>=2.4.0  # For DICOM support
```

---

## 📖 Usage

### **1. Inference on Single Image**

```python
from models import CMTUnetLarge
from utils import load_image, visualize_results

# Load model
model = CMTUnetLarge.from_pretrained("checkpoints/cmt-unet-large.pth")
model.eval()
model.cuda()

# Load and preprocess image
image = load_image("data/test/frame_001.png", size=(1024, 1024))

# Inference
with torch.no_grad():
    mask, class_logits = model(image)

# Visualize results
visualize_results(image, mask, class_logits, save_path="output/result.png")
```

### **2. Batch Inference on Video**

```python
from models import CMTUnetSmall
from utils import VideoProcessor

# Initialize model
model = CMTUnetSmall.from_pretrained("checkpoints/cmt-unet-small.pth")

# Process video
processor = VideoProcessor(model)
processor.process_video(
    input_path="data/videos/surgery.mp4",
    output_path="output/segmented_video.mp4",
    fps=30
)
```

### **3. Training from Scratch**

```bash
# Stage 1: Encoder Alignment (Feature Distillation)
python train.py \
    --config configs/encoder_alignment.yaml \
    --teacher sam_vit_h \
    --student cmt-unet-large \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4

# Stage 2: Decoder Alignment (Mask Distillation)
python train.py \
    --config configs/decoder_alignment.yaml \
    --checkpoint checkpoints/encoder_aligned.pth \
    --batch_size 16 \
    --epochs 50

# Stage 3: Supervised Fine-tuning (End-to-End)
python train.py \
    --config configs/fine_tuning.yaml \
    --checkpoint checkpoints/decoder_aligned.pth \
    --dataset miccai+cholecseg8k \
    --batch_size 8 \
    --epochs 50 \
    --augmentation strong
```

### **4. Evaluation**

```bash
# Test on MICCAI dataset
python test.py \
    --checkpoint checkpoints/cmt-unet-large.pth \
    --dataset miccai \
    --split test \
    --save_masks \
    --save_metrics results/metrics.csv

# Benchmark inference speed
python benchmark.py \
    --model cmt-unet-small \
    --num_runs 100 \
    --device cuda \
    --precision fp16
```

---

## 📦 Datasets

### **MICCAI EndoVis 2017**
- **Description**: Robotic Instrument Segmentation Challenge
- **Source**: da Vinci® Xi robot, porcine surgeries
- **Size**: 8 video sequences, 1800 frames total
- **Resolution**: 1920×1080 → resized to 1024×1024
- **Instruments**: 7 classes
  - Large needle drivers
  - ProGrasp forceps
  - Bipolar forceps
  - Grasping retractors
  - Maryland bipolar forceps
  - Monopolar curved scissors
  - Vessel sealers

**Download**: [MICCAI EndoVis Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

### **CholecSeg8k**
- **Description**: Laparoscopic cholecystectomy segmentation
- **Source**: 80 laparoscopic videos
- **Size**: 8000 annotated frames
- **Resolution**: Variable → resized to 1024×1024
- **Usage**: Fine-tuning and generalization testing

**Download**: [CholecSeg8k Dataset](https://github.com/CAMMA-public/cholec80)

### **Data Preparation**

```bash
# Download and prepare MICCAI dataset
python scripts/prepare_miccai.py \
    --download_dir data/raw \
    --output_dir data/processed/miccai \
    --resize 1024

# Download and prepare CholecSeg8k
python scripts/prepare_cholecseg.py \
    --download_dir data/raw \
    --output_dir data/processed/cholecseg8k \
    --split 0.8
```

---

## 🔬 Training Strategy

### **Multi-Stage Knowledge Distillation**

#### **Stage 1: Encoder Alignment (Pre-training)**
```python
Loss = MSE(f_teacher, f_student)
```
- **Goal**: Align intermediate feature embeddings
- **Frozen**: Teacher encoder, Student decoder
- **Trainable**: Student encoder
- **Duration**: 50 epochs with early stopping

#### **Stage 2: Decoder Alignment (Pre-training)**
```python
Loss = 0.5 × Dice + 0.5 × BCE
```
- **Goal**: Learn to generate binary masks
- **Frozen**: Both encoders
- **Trainable**: Student decoder
- **Duration**: 50 epochs with early stopping

#### **Stage 3: Supervised Fine-tuning**
```python
Loss_seg = 0.5 × Dice + 0.5 × BCE
Loss_cls = CrossEntropy
Loss_total = Loss_seg + Loss_cls
```
- **Goal**: Refine segmentation + learn classification
- **Frozen**: None (end-to-end training)
- **Trainable**: Entire network + classification head
- **Duration**: 50 epochs with CutMix/MixUp augmentation

---

## 📈 Experiment Tracking

We use **Weights & Biases** for experiment tracking:

```python
import wandb

wandb.init(
    project="surgical-tool-segmentation",
    config={
        "model": "cmt-unet-large",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "epochs": 50
    }
)

# Log metrics
wandb.log({
    "train/loss": train_loss,
    "val/iou": val_iou,
    "val/dice": val_dice
})
```

View our experiments: [W&B Dashboard](https://wandb.ai/mthezn/surgical-tool-segmentation)

---

## 🎨 Qualitative Results

### **Challenging Scenarios**

Our model handles:
- ✅ **Motion blur** from rapid instrument movements
- ✅ **Overlapping tools** (instrument collision)
- ✅ **Over-illumination** and specular reflections
- ✅ **Extreme zoom** (close-up views)
- ✅ **Blood and tissue occlusion**

Example outputs:

```
data/
├── examples/
│   ├── motion_blur.png       # CMT-Unet handles motion blur
│   ├── overlap.png           # Separates overlapping instruments
│   ├── illumination.png      # Robust to lighting changes
│   └── zoom.png              # Works at various zoom levels
```

---

## 🏥 Clinical Applications

### **Potential Use Cases**

1. **Real-time Surgical Guidance**
   - Instrument tracking during surgery
   - Collision detection and warning systems

2. **Surgical Skill Assessment**
   - Automatic instrument usage analysis
   - Performance metrics extraction

3. **Autonomous Surgery**
   - Robotic path planning
   - Visual servoing for robotic instruments

4. **Post-operative Analysis**
   - Video review and annotation
   - Surgical workflow analysis

---

## 📊 Model Zoo

| Model | Params | FLOPs | Latency (ms) | mIoU | Download |
|-------|--------|-------|--------------|------|----------|
| CMT-Unet-Large | 320M | 42G | 21.78 | 0.8919 | [weights](https://drive.google.com/...) |
| CMT-Unet-Small | 78M | 34G | 15.48 | 0.8703 | [weights](https://drive.google.com/...) |
| CMT-Unet-Large + Cls | 325M | 43G | 50.12 | 0.8919 | [weights](https://drive.google.com/...) |

---

## 🛠️ Deployment

### **ONNX Export**

```python
import torch
from models import CMTUnetLarge

model = CMTUnetLarge.from_pretrained("checkpoints/cmt-unet-large.pth")
model.eval()

dummy_input = torch.randn(1, 3, 1024, 1024)

torch.onnx.export(
    model,
    dummy_input,
    "models/cmt-unet-large.onnx",
    export_params=True,
    opset_version=14,
    input_names=['image'],
    output_names=['mask', 'class_logits']
)
```

### **TensorRT Optimization**

```bash
# Convert ONNX to TensorRT
trtexec \
    --onnx=models/cmt-unet-large.onnx \
    --saveEngine=models/cmt-unet-large.trt \
    --fp16 \
    --workspace=4096

# Benchmark TensorRT engine
trtexec --loadEngine=models/cmt-unet-large.trt --iterations=100
```

---

## 🔍 Ablation Studies

### **Impact of Distillation Stages**

| Configuration | mIoU | Dice | Latency (ms) |
|--------------|------|------|--------------|
| Encoder-only distillation | 0.8893 | 0.9335 | 24.80 |
| Encoder + Decoder distillation | **0.8919** | **0.9355** | **21.78** |

**Finding**: Dual distillation improves both quality and speed.

### **Effect of Decoder Architecture**

| Decoder | mIoU | Parameters | FLOPs |
|---------|------|------------|-------|
| Transformer-based | 0.8856 | 380M | 58G |
| **CNN-based (ours)** | **0.8919** | **320M** | **42G** |

**Finding**: CNN decoder offers better efficiency-accuracy trade-off.

---

## 🐛 Known Limitations

1. **Sequential Pipeline**: Segmentation and classification are performed in two stages, increasing latency.
2. **No Temporal Reasoning**: Each frame is processed independently (no video continuity).
3. **Dataset Coverage**: Trained primarily on da Vinci® robot data; generalization to other systems needs validation.
4. **Class Imbalance**: Vessel sealer class has lower recall (0.5274) due to limited training samples.

---

## 🔮 Future Work

- [ ] **Single-Stage Architecture**: Merge segmentation and classification into one forward pass
- [ ] **Temporal Integration**: Add LSTM/Transformer for video sequence modeling
- [ ] **Multi-View Fusion**: Leverage stereo endoscope data
- [ ] **Active Learning**: Improve rare class performance with targeted data collection
- [ ] **Real-time Deployment**: Optimize for surgical robot onboard computers
- [ ] **3D Reconstruction**: Extend to 3D pose estimation

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{dezen2025surgical,
  title={Efficient Prompt-Free Surgical Tools Segmentation and Classification via Knowledge Distillation},
  author={De Zen, Marco},
  year={2025},
  school={Politecnico di Milano},
  type={Master's Thesis},
  note={Submitted to IEEE Transactions on Medical Imaging}
}
```

---

## 👥 Contributors

- **Marco De Zen** - Main Author ([GitHub](https://github.com/mthezn) | [LinkedIn](https://linkedin.com/in/marcodezen))
- **Prof. Elena De Momi** - Supervisor
- **Mattia Magro** - Co-advisor

---

## 📧 Contact

For questions, collaborations, or issues:

- 📧 Email: marco.dezen01@gmail.com
- 💼 LinkedIn: [linkedin.com/in/marcodezen](https://linkedin.com/in/marcodezen)
- 🐙 GitHub: [github.com/mthezn](https://github.com/mthezn)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Meta AI** for the Segment Anything Model (SAM)
- **MICCAI EndoVis Challenge** organizers
- **CholecSeg8k** dataset contributors
- **Politecnico di Milano** - NearLab research group

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=mthezn/surgical-tool-segmentation&type=Date)](https://star-history.com/#mthezn/surgical-tool-segmentation&Date)

---

<div align="center">

**Made with ❤️ for advancing surgical robotics**

[Report Bug](https://github.com/mthezn/surgical-tool-segmentation/issues) · [Request Feature](https://github.com/mthezn/surgical-tool-segmentation/issues)

</div>
