import torch
from models.time_transformer import TimeTransformer, TimeTransformerConfig

def main():
    B, K, D = 2, 10, 768
    x = torch.randn(B, K, D)

    cfg = TimeTransformerConfig(
        input_dim=D,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
    )

    model = TimeTransformer(cfg)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    main()
