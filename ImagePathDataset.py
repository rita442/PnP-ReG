from torch.utils.data import Dataset
from PIL import Image


class ImagePathDataset(Dataset):
    def __init__(self, list_paths, transform=None, virtual_length=0):
        self.list_paths=list_paths
        self.transform = transform
        real_length = len(self.list_paths)
        if virtual_length:
            self.virtual_length = virtual_length
        else:
            self.virtual_length = real_length
        self.modulo_length = min(real_length, self.virtual_length)

    def __getitem__(self, i):
        img = Image.open(self.list_paths[i % self.modulo_length])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.virtual_length

