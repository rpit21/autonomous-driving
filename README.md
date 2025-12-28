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
│── models/
│   ├── vit_encoder.py
│   └── temporal_transformer.py
│
│── dataset/
│   └── bddv_dataset.py
│
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

## Dataset objetive
Its main goal is to convert "raw videos" in to samples that contains:

- Inputs: temporal windows with K consecutives frames and B batch size (clips in paralle) `[B, K, RGB:3, H, W]`
- Label (target): the "next" action (steering, aceleration)


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

## VIT Encoder functionality

The data set give: `frames: [B, K, 3, 244, 244]`

But for the "Temporal transformer"  it needs vectors. That is why it is necesarry a VIT encoder wich:

- Each frame -> VIT -> output: 1 summary vector (CLS Token)
- K frames -> K vectors/tokens

Expected Output:

`[B, K, D]`

> D is the size of the embedding of VIT. Remeber K is the time window, B batch size

> The reason why we reshape `[B, K, 3, H, W] → [B*K, 3, H, W]` is to make the VIT to process al the images together

---
Freeze On

Makes the VIT to not be trained

>In case of training the VIT you can put it in unfreeze mode

---
The VIT model that we are using has this characteristics

| Property            | Value   |
| ------------------- | ------- |
| Patch size          | 16×16   |
| Image               | 224×224 |
| Embedding dim (`D`) | **768** |
| Nº layer            | 12      |
| Nº heads            | 12      |


---
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

## Time transformer
Its objective it is to:

- To take a sequences of temporal states
- Create a final representation of the actual time state

```Mathematica
Input:  [B, K, D]   ← embeddings por frame (CLS + sensores luego)
Output: [B, H]      ← embedding temporal final
```

1-2 Layer

Multi-head self-attention

Output: Last temporal token

## MLP Objective

It is the one who generates the final desition

- Input: time transforme output
- Output: actions `[a_steer, a_accel]`

## Driving Model

1. Take a batch form the real data set (frames are `[B,K,3,224,224]`)

2. The model make:
    - ViTFrameEncoder `frames -> [B,K,768]` (selecting only CLS token)
    - Concatenate sensor (OFF right now) `use_sensors=False`
    - Time Transformer (`[B,K,768] -> [B,256]` actual temporal state)
    - MLP (`[B,256] -> [B,2]` stearing and acceleration)


## Training 
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


## What I need to do
Checklist to see if we are ready to work:

✅ Repo creado

✅ Python + venv

✅ Transformers instalados

✅ Dataset descargándose

✅ Dataset loader propio
✅ Decisión final de targets (stearing wheel and aceleration)
⬜ Arquitectura definida en código (no idea)
⬜ Reparto de tareas
⬜ README mínimo (In progresss)
