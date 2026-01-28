# End-to-End Autonomous Driving with Transformers

This project implements an end-to-end imitation learning system for autonomous driving.
The model predicts continuous steering angle and acceleration directly from driving videos
using Vision Transformers and temporal attention mechanisms.

Final grade: **30/30**

## Qualitative Demo

The following animation shows the predicted steering angle and acceleration
overlaid on a real driving video sequence.

![Driving demo](figures/demo.gif)

## Architecture Overview

The model is composed of three main blocks:

- **Vision Transformer (ViT)**  
  Extracts high-level visual features from each video frame using a pre-trained and frozen backbone.

- **Temporal Transformer**  
  Models short-term driving dynamics over a fixed temporal window of K frames.

- **MLP Regression Head**  
  Predicts continuous steering and acceleration commands.


## Results

The model was trained and evaluated on the Berkeley DeepDrive Video (BDDV) dataset.

- Continuous regression of steering and acceleration
- Smooth and temporally consistent predictions
- Evaluated using MAE and RMSE metrics

See the report for full quantitative and qualitative evaluation.


## Structure of the project

```javascript
project/
│── data/
│   └── bddv_subset/
│
│── dataset/
│   └── bddv_dataset.py
│
│── models/
│   ├── vit_encoder.py
│   ├── time_transformer.py
│   ├── mlp_head.py
│   └── driving_model.py
│
│── train_one_step.py
│── train.py
│── eval.py
│── requirements.txt
│── README.md

```

## Installation

Python 3.10 is required.

Create and activate a virtual environment:


```bash
python -m venv .venv
```
Activate:

```bash
.venv\Scripts\activate
```

Install libraries:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install transformers timm
python -m pip install numpy matplotlib tqdm
python -m pip install opencv-python
```

Another way to install all the libraries used in the proyect use this comand:

```bash
pip install -r requirements.txt
```

## Dataset
The project uses a custom temporal dataset built on top of BDD100K videos (BDDV).

Each sample consists of:
- K consecutive RGB frames
- Synchronized vehicle sensor data
- Continuous control targets: steering and acceleration

The dataset implementation can be found in:
`dataset/bddv_dataset.py`

## Training
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Temporal window: K = 10 frames
- Regression task (steering, acceleration)

Training and evaluation scripts:
- `train.py`
- `eval.py`


## Team
- Dataset & Sensors
- Model & Training
- Evaluation & Analysis
