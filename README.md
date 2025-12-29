# Autonomous-Driving

## Pipeline

```scss
Frames
 → ViT
 → CLS token per frame
 → (concatenate sensors per frame)  
 → Time Transformer
 → ft (estado presente)
 → MLP
 → steering / accel


```

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

## Requeriments

Install Python 3.10.11

### Virtual enviroment

Create:

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

## Dataset objetive (bddv_dataset.py)
Its main goal is to convert "raw videos" wiht its information in to samples that contains:

- Inputs: temporal windows with K consecutives frames and B batch size (clips in paralle) `[B, K, RGB:3, H, W]`
- Sensors: temporal window of K consecutives sensors information `[B, K, S]`
- Targets: the action taken (steering, aceleration) `[2]`


### Dataset Pipline:
```rust
bddv_dataset.py
└── __getitem__(idx)
    ├── Read video → frames [K,3,224,224]
    ├── Read info.json → sensors [K,S]
    ├── Calculate target → targets [2]
    └── return frames, sensors, targets

#Batching logic outside the dataset

DataLoader
└── Gather B samples
    └── batch = (frames[B,...], sensors[B,...], targets[B,...])
```

> Dataset: defines the input of the model 
## Dataset: BDDV Temporal Driving Dataset

### Overview

This project uses a custom dataset built on top of BDD100K driving videos, referred to as **BDDV** (BDD Video).
The dataset converts raw driving videos and their associated metadata into **temporally consistent samples**
suitable for end-to-end autonomous driving models.

Each dataset sample represents a short driving sequence and contains:
- A temporal window of consecutive RGB frames
- Synchronized vehicle sensor data
- The corresponding continuous driving action

The dataset implementation is located in:

dataset/bddv_dataset.py
---

### Dataset Objective

The objective of the dataset is to enable learning of the following mapping:
(visual observations over time, sensor states over time) → driving action at current time

Rather than using single images, the dataset provides **fixed-length temporal windows** to capture motion,
vehicle dynamics, and delayed control effects.

---

### Sample Structure

For each dataset index, the dataset returns:

| Component | Shape | Description |
|---------|------|-------------|
| `frames` | `[K, 3, 224, 224]` | K consecutive RGB frames |
| `sensors` | `[K, S]` | Sensor values aligned with each frame (optional) |
| `targets` | `[2]` | Continuous control targets `[steering, acceleration]` |

After batching with a PyTorch `DataLoader`:

| Component | Shape |
|---------|------|
| `frames` | `[B, K, 3, 224, 224]` |
| `sensors` | `[B, K, S]` |
| `targets` | `[B, 2]` |

Where:
- **B** = batch size  
- **K** = temporal window length  
- **S** = number of sensor channels  

---

### Dataset Pipeline

The dataset follows the pipeline below to generate each sample:


---

### `__getitem__` Logic

```text
bddv_dataset.py
└── __getitem__(idx)
    ├── Load video corresponding to idx
    ├── Extract K consecutive frames
    ├── Resize frames to 224×224
    ├── Load sensor data from info.json
    ├── Compute target [steering, acceleration]
    └── Return (frames, sensors, targets)
### Dataset Semantics and Usage

Each call to `__getitem__` returns **one temporal sample**.
Batching and shuffling are handled externally by PyTorch’s `DataLoader`.

---

### Temporal Alignment

All components within a sample are **temporally aligned**:

- Frame *t* corresponds to sensor values at time *t*
- The target represents the control action taken at time *t*

This alignment ensures the model learns **causal temporal relationships**
rather than frame-wise correlations.

---

### Why Temporal Windows?

Using temporal windows instead of single frames enables the model to learn:

- Motion and scene dynamics
- Vehicle behavior over time
- Smooth and stable control actions

This is essential for predicting **continuous steering and acceleration**
in end-to-end driving models.

---

### Design Principles

- Video-based temporal input instead of single images
- Fixed-length temporal windows for efficient batching
- Continuous control prediction (regression)
- Clear separation of responsibilities:
  - **Dataset** → data preparation
  - **Model** → representation learning
  - **Training** → optimization

---

### Dataset Responsibilities

**Handled by the dataset**
- Video loading and decoding
- Temporal frame slicing
- Sensor extraction and alignment
- Target computation

**Not handled by the dataset**
- Batching and shuffling
- Device transfer (CPU/GPU)
- Temporal modeling
- Loss computation

---

### Intended Usage

```python
dataset = BDDVDataset(cfg)
loader = DataLoader(dataset, batch_size=B, shuffle=True)

for frames, sensors, targets in loader:
    preds = model(frames, sensors)
    loss = criterion(preds, targets)







## VIT Encoder functionality (vit_encoder.py)

The data set give: `frames: [B, K, 3, 244, 244]`

But for the "Temporal transformer"  it needs vectors. That is why it is necesarry a VIT encoder wich:

- Each frame -> VIT -> output: 1 summary vector (CLS Token)
- K frames -> K vectors/tokens

Expected Output:

`[B, K, D]`

> D is the size of the embedding of VIT. Remeber K is the time window, B batch size

> The reason why we reshape `[B, K, 3, H, W] → [B*K, 3, H, W]` is to make the VIT to process al the images together

Sumary of VIT data:

```
Input:
[B, K, 3, 224, 224]

Reshape:
[B*K, 3, 224, 224]

ViT:
[B*K, D]

Reshape:
[B, K, D]  ≡  [st-k, ..., st]

```

---
### Freeze Mode

**Freeze ON** --> Makes the VIT to not be trained

>In case of training the VIT you can put it in *unfreeze mode*

---
### VIT model characteristics

| Property            | Value   |
| ------------------- | ------- |
| Patch size          | 16×16   |
| Image               | 224×224 |
| Embedding dim (`D`) | **768** |
| Nº layer            | 12      |
| Nº heads            | 12      |


---
### VIT PIP line logic 

The logic For only one frame: 

```Mathematica
Image
↓
Divide in patches → N patches
↓
Embedding each patch → N tokens
↓
Include 1 extra token → [CLS]
↓
Total tokens = N + 1
↓
Transformer (VIT)
↓
Output = N + 1 tokens
```


## Time transformer (time_transformer.py)
Its objective it is to:

- To take a sequences of temporal states
- Create a final representation of the actual time state

```Mathematica
Input:  [B, K, D]   ← embeddings por frame (CLS + sensors)
Output: [B, H]      ← embedding temporal final
```

1-2 Layer

Multi-head self-attention

Output: Last temporal token

## MLP Objective (mpl_head.py)

It is the one who generates the final desition

- Input: time transforme output
- Output: actions `[a_steer, a_accel]`

## Driving Model (driving_model.py)

1. Take a batch form the real data set (frames are `[B,K,3,224,224]`)

2. The model make:
    - ViTFrameEncoder `frames -> [B,K,768]` (selecting only CLS token)
    - Concatenate sensor (OFF right now) `use_sensors=False`
    - Time Transformer (`[B,K,768] -> [B,256]` actual temporal state)
    - MLP (`[B,256] -> [B,2]` stearing and acceleration)


## Training (train.py)
As we are trying to predict continuous values we are going to use:

Mean Squared Error Loss:
- Check how much you get wrong `y - ŷ`
- Increse at square (penalize big errors)
- Make the mean

> It is not usede Cross-entropy because is for clasification
> MAE (Mean Absolute error) is an alternative option but it is less strong

It is used ADAM because it is an optimizer way to make Gradient descendant
- Remember past gradients and avoid zig zags
- Each parameter has its own learning rate

The training loop will make:
1. `y_pred = model(frames)`
2. `loss = MSE(y_pred, y_true)`
3. `loss.backward()` which is the gradient
4. `optimizer.step()` Adam update weights

```pgsql
videos/ (raw .mov)
   |
   v
BDDVDataset.__getitem__(idx)
   |
   |-- reads K frames -> tensor [K,3,224,224]
   |-- returns (frames, info)
   v
DataLoader (batching)
   |
   |-- stacks B items -> frames [B,K,3,224,224]
   v
Training loop
   |
   |-- frames.to(device)
   |-- preds = model(frames)
   |       |
   |       |-- ViT: [B,K,3,224,224] -> [B,K,768]
   |       |-- TimeTransformer: [B,K,768] -> [B,256]
   |       |-- MLP: [B,256] -> [B,2]
   |
   |-- loss = MSE(preds, targets)
   |-- backward()
   |-- optimizer.step()
   v
weights updated

```

```rust
dataset[idx] -> one sample
      |
DataLoader (batch_size=B)
      |
batch = tuple of B samples
      |
unpack_batch(batch)
      |
frames   [B,K,3,224,224]
sensors  [B,K,S] or None
targets  [B,2]
      |
model(frames, sensors)
      |
preds [B,2]
      |
loss(preds, targets)
      |
backward + optimizer.step

```
### Intentional Overfitting 
Overfitting means = the model memorize all the train data
- This doesn't generalize
- Don't work with new data

But, also means:

The model has the sufficient capacity and it is well conected.

> If the model cannot be overfit the model is not working properly it.

To make intentional Overfitting:
Train loss = 0

**Results:** It generates a overfit --> model is capable to learn

> My modelo end-to-end is trainable and able to memorize a visual sequence.

### Save of the weights - Checkpoints
Normally all the train data is save on the RAM but when the script ends it desappear.

For that is importat to save all the weights in a file 

```py
torch.save(model.state_dict(), "model.pt")
```

Normally it has to be saved at the final of the training 

To load the weight you can use:

```py
model = DrivingModel(cfg)
model.load_state_dict(torch.load("overfit_model.pt"))
model.eval()
```

### Train Indications
- Save check points
- Implementation of debug Mode


## Team workflow

The project is divided into three independent modules:

1. Dataset & Sensors
   - Video loading
   - Sensor extraction and normalization
   - Target generation

2. Model & Training
   - ViT + Temporal Transformer + MLP
   - Training loop
   - Debugging

3. Evaluation
   - MAE / RMSE metrics
   - Plots and tables
   - Final report
