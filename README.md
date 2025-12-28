# Autonomous-Driving

## Pipeline

```scss
video BVDD
    - frames (k consecutives)
    - sensor (velocity, angle)
        |
viT (for frames)
        |
Token temporales (K)
        |
Temporal transformer
        |
Action prediction 

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

## VIT function

For only one frame 

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

## What I need to do
Checklist to see if we are ready to work:

✅ Repo creado

✅ Python + venv

✅ Transformers instalados

✅ Dataset descargándose

⬜ Dataset loader propio
⬜ Decisión final de targets (stearing wheel and aceleration)
⬜ Arquitectura definida en código (no idead)
⬜ Reparto de tareas
⬜ README mínimo (In progresss)
