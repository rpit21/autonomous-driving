from torch.utils.data import DataLoader
from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig

def main():
    cfg = BDDVDatasetConfig(
        root_dir="data/bddv/videos",
        clip_len=10,      # K frames
        stride=10,        # less clips for debug
        frame_step=1,
        resize_hw=(224, 224),
    )

    dataset = BDDVDataset(cfg) 
    print(f"Dataset length: {len(dataset)} clips")  #total number of clips from all the data videos
                                                    # Each clip is a point with 10 temporal frames

    loader = DataLoader(
        dataset,
        batch_size=2,    # 2 clips in parallel
        shuffle=True,
        num_workers=0,   # important on Windows
    )

    frames, info = next(iter(loader))

    print("Frames shape:", frames.shape) # batch_size, k frames for clip, RGB channels, size for frame
    print("Frames dtype:", frames.dtype)
    print("Frames min/max:", frames.min().item(), frames.max().item()) #Normalized

    print("\nSample info:")
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
