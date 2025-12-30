import torch
from models.mlp_head import MLPHead, MLPHeadConfig

def main():
    B = 2
    H = 256  # output dim of TimeTransformer

    x = torch.randn(B, H)

    cfg = MLPHeadConfig(
        input_dim=H,
        hidden_dim=128,
        output_dim=2,   # steering + accel
    )

    mlp = MLPHead(cfg)
    y = mlp(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output:", y)

if __name__ == "__main__":
    main()
