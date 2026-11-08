import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .charset import Charset

class HTRDataset(Dataset):
    def __init__(self, csv, charset_path, split, height=64, max_width=1024, transform=None):
        df = pd.read_csv(csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.charset = Charset.load(charset_path)
        self.h, self.maxw, self.transform = height, max_width, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)               # аугментации позже
        w = min(max(1, round(img.shape[1] * self.h / img.shape[0])), self.maxw)
        img = cv2.resize(img, (w, self.h), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 255 - 0.5) / 0.5          # [-1, 1], фон ≈ +1
        target = torch.tensor(self.charset.encode(r.text), dtype=torch.long)
        return torch.from_numpy(img)[None], target                # (1, H, W), (L,)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    image_lengths = torch.tensor([im.shape[-1] for im in imgs], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    B, H = len(imgs), imgs[0].shape[1]
    images = torch.ones(B, 1, H, int(image_lengths.max()))        # паддинг белым
    padded = torch.zeros(B, int(target_lengths.max()), dtype=torch.long)
    for k, (im, t) in enumerate(zip(imgs, targets)):
        images[k, :, :, :im.shape[-1]] = im
        padded[k, :len(t)] = t
    return images, padded, target_lengths, image_lengths