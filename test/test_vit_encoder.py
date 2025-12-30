import torch
from torch.utils.data import DataLoader

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.vit_encoder import ViTFrameEncoder, ViTEncoderConfig

def main():
    # 1) Load dataset
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/videos",
        clip_len=10,
        stride=10,
        frame_step=1,
        resize_hw=(224, 224),
    )
    dataset = BDDVDataset(ds_cfg)

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    frames, info = next(iter(loader))

    print("Input frames:", frames.shape)  # [B,K,3,224,224]

    # 2) Create ViT encoder
    vit_cfg = ViTEncoderConfig(
        model_name="google/vit-base-patch16-224-in21k",
        freeze=True,
        input_is_0_1=True,  # your dataset outputs [0,1]
    )
    encoder = ViTFrameEncoder(vit_cfg)

    # 3) Forward
    with torch.no_grad():
        feats = encoder(frames)

    print("Output features:", feats.shape)  # [B,K,D]
    print("D (hidden size):", encoder.hidden_size)

if __name__ == "__main__":
    main()
