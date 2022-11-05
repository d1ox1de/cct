import torch
import cv2
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Iterator, List, Optional, Any, Tuple, Dict

from .augmentation import augment_hsv, img_rotate_angle


torch.manual_seed(0)
np.random.seed(0)

# All pre-trained torch models  expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
# where H and W are expected to be at least 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENTIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

class BalancedSampler:

    def __init__(self, dataset: Dataset, num_oversamples: Optional[List] = None, replacement: bool = True, augmentation_prefix: str = 'aug'):
        self.filenames = dataset.filenames
        self.labels = dataset.labels
        self.label_names = list(dataset.class_labels.values())
        self.replacement = replacement
        self.augmentation_prefix = augmentation_prefix
        num_oversamples = np.array(num_oversamples) if  num_oversamples is not None else np.zeros(self.labels.shape[1])

        self.num_samples = np.sum(self.labels, axis=0)
        self.target_numbers = np.rint(np.sum(self.labels, axis=0) + num_oversamples)

        # There might be augmented images (those images contain augmentation_prefix in their paths) in the dataset.
        # no need to oversample them
        aug_files = np.array([augmentation_prefix in str(filename, encoding='utf-8') for filename in self.filenames])
        self.num_samples_aug = np.sum(self.labels[aug_files], axis=0, dtype=np.float64)
        self.weights = self._get_weights()

        # prints an approximate number of images of each class after oversampling (undersampling)
        self._print_resulting_dataset()
        return None


    def _get_weights(self):
        weights = [0] * self.labels.shape[0]

        class_weights_aug = np.ones(self.num_samples_aug.shape[0], dtype=np.float64) / np.sum(self.target_numbers)
        class_weights_true = (abs(self.target_numbers - self.num_samples_aug) / (1e-16 + self.num_samples - self.num_samples_aug)) / np.sum(self.target_numbers, dtype=np.float64)
        for i, (filename, label) in enumerate(zip(self.filenames, self.labels)):
            label = label.astype(bool)
            if self.augmentation_prefix in str(filename, encoding='utf-8'):
                class_weights = class_weights_aug[label]
            else:
                class_weights = class_weights_true[label]
            weights[i] = np.mean(class_weights)
        return weights


    def _print_resulting_dataset(self) -> None:
        print('\nLabel: Num genuine imgs / Total imgs --> Num genuine imgs oversampled / Total imgs oversampled\n')

        for i, (label_name, init_number, target_number) in enumerate(zip(self.label_names, self.num_samples, self.target_numbers)):
            true_init_number = init_number - self.num_samples_aug[i]
            target_true_number = target_number - self.num_samples_aug[i]
            print(f'{label_name:>10}:  {true_init_number:>5.0f}/{init_number:>5.0f} --> {target_true_number:>5.0f}/{target_number:>5.0f}')
        return None


    def __iter__(self) -> Iterator[int]:
        weights = torch.as_tensor(self.weights, dtype=torch.float64)
        rand_tensor = torch.multinomial(weights, self.__len__(), self.replacement)
        yield from iter(rand_tensor.tolist())


    def __len__(self) -> int:
        return np.sum(self.target_numbers, dtype=np.int32)


class CustomDataset(Dataset):
    # https://github.com/pytorch/pytorch/issues/13246
    def __init__(self, csv: str, hyp: Dict[str, float], is_augment: bool = False, img_size: int = 224) -> None:
        df = pd.read_csv(csv)
        #print stats
        print(df[df.columns[1:]].sum())

        self.filenames = np.array(df['filename']).astype(np.string_) # dtype='|S5';  dtype='<U20' is not a square byte array i.e. not a single object
        # label: 0/1 one-dimensional array of size N_LABELS
        self.labels = np.array(df.drop(['filename'], axis=1)).astype(np.float32)
        self.num_classes = self.labels.shape[1]
        self.class_labels = dict(zip(range(self.num_classes), df.columns[1:])) # {0: "no_object", 1: "label1", ...}
        self.img_size = img_size
        self.hyp = hyp
        self.is_augment = is_augment

        self.normalize = transforms.Normalize(mean=MEAN, std=STD)


    def __len__(self) -> int:
        return self.filenames.shape[0]


    def get_num_classes(self):
        return self.labels.shape[1]


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(str(self.filenames[index], encoding='utf-8')) # BGR, (H, W, 3)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        labels = self.labels[index]

        if self.is_augment:
            #it's better to use torch rand operations
            augment_hsv(img, self.hyp['hsv_h'], self.hyp['hsv_s'], self.hyp['hsv_v'])

            if torch.rand(1).item() < self.hyp['flip_lr']:
                img = np.fliplr(img)

            if torch.rand(1).item() < self.hyp['flip_ud']:
                img = np.flipud(img)

            img = img_rotate_angle(img, self.hyp['angle_max'])

            # img = img_rotate_multipleof90(img, torch.randint(4, size=(1,)).item())

        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR->RGB, (H, W, 3) -> (3, H, W)
        img = img.astype(np.float32)

        img = torch.from_numpy(img) / 255.
        labels = torch.from_numpy(labels)

        #should be torch tensor of float dtype
        img = self.normalize(img)
        return img, labels


def create_dataset_dataloader(
        csv: str,
        hyp: Dict[str, Any],
        batch_size: int,
        is_augment: bool = False,
        shuffle: bool = False,
        world_size: int = 1,
        is_balanced_sampler: bool = False) -> Tuple[DataLoader, CustomDataset]:
    # Also, once you pin a tensor or storage, you can use asynchronous GPU copies.
    # Just pass an additional non_blocking=True argument to a to() or a cuda() call
    sampler = None
    dataset = CustomDataset(str(Path(csv)), hyp, is_augment=is_augment)
    if hyp["num_oversamples"] is not None and is_balanced_sampler:
        sampler = BalancedSampler(dataset, hyp["num_oversamples"])
        shuffle = False # sampler option is mutually exclusive with shuffle
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw, pin_memory=True, drop_last=True, sampler=sampler)

    return dataloader, dataset


class DatasetInference(Dataset):
    def __init__(self, input_dir: str, img_size: int = 224) -> None:
        self.img_paths = self._get_imgs_paths(input_dir)
        self.img_size = img_size

        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def _get_imgs_paths(self, input_dir: str) -> np.ndarray:
        img_paths = []
        for img_p in Path(input_dir).rglob('*'):
            if img_p.suffix not in EXTENTIONS:
                continue
            img_paths.append(str(img_p))
        return np.array(img_paths).astype(np.string_)

    def __len__(self) -> int:
        return self.img_paths.shape[0]

    def __getitem__(self, index) -> Tuple[str, torch.Tensor]:
        img_path = str(self.img_paths[index], encoding='utf-8')
        try:
            img = cv2.imread(img_path) # BGR, (H, W, 3)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
            return None, None

        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR->RGB, (H, W, 3) -> (3, H, W)
        # img = np.ascontiguousarray(img)
        img = img.astype(np.float32)

        img = torch.from_numpy(img) / 255.

        #should be torch tensor of float dtype
        img = self.normalize(img)
        return img.unsqueeze(0), img_path

    @staticmethod
    def collate_fn(batch: List[Tuple[Optional[str], Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, List[str]]:
        img_paths = []
        imgs = []
        for img, p in batch:
            if p is None:
                continue
            img_paths.append(p)
            imgs.append(img)

        return torch.cat(imgs, dim=0), img_paths


def create_dataloader(input_dir: str, batch_size: int, num_workers: Optional[int] = None) -> DataLoader:
    # Also, once you pin a tensor or storage, you can use asynchronous GPU copies.
    # Just pass an additional non_blocking=True argument to a to() or a cuda() call
    dataset = DatasetInference(input_dir)
    num_workers = num_workers if num_workers is not None else min([os.cpu_count()//2, batch_size])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=DatasetInference.collate_fn)
    return dataloader

def test():
    hyp = {
        'flip_lr': 0.5,
        'flip_ud': 0.5,
        "num_oversamples": [1., 200., -100., -5000, 490]
    }
    csv = Path(r'..\..\csv\train.csv')
    dataset = CustomDataset(str(csv), hyp, is_augment=True)
    print(dataset.filenames.shape)
    sampler = BalancedSampler(dataset, hyp['num_oversamples'])
    for _ in range(5):
        samples = []
        for _, (sampl) in enumerate(sampler):
            samples.append(sampl)
        samples = np.hstack(samples)
        labels = dataset.labels[samples]
        print('custom', np.sum(labels, axis=0), np.sum(labels))

    return None

if __name__ == "__main__":
    test()