import torch
from torch.utils.data import DataLoader

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


def main():
    # 1) Load a real batch from your dataset
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
    print("Frames:", frames.shape)

    # 2) Build the full driving model (vision-only for now)
    model_cfg = DrivingModelConfig(
        predict_dim=2,  # steering + accel
        use_sensors=False,  # we'll add sensors next
        sensor_dim=0,
        vit=ViTEncoderConfig(
            model_name="google/vit-base-patch16-224-in21k",
            freeze=True,
            input_is_0_1=True,
        ),
    )

    model = DrivingModel(model_cfg)

    # 3) Forward pass (no training yet)
    with torch.no_grad():
        y = model(frames)

    print("Model output:", y.shape)
    print(y)


if __name__ == "__main__":
    main()
