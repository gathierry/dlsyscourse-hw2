import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        h, w, c = img.shape
        img_pad = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        shift_x += self.padding
        shift_y += self.padding
        img = img_pad[shift_x : shift_x + h, shift_y : shift_y + w]
        return img


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        inds = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(inds)
        self.ordering = iter(np.array_split(
            inds, range(self.batch_size, len(inds), self.batch_size)
        ))
        return self

    def __next__(self):
        inds = next(self.ordering)
        data = []
        for i in inds:
            data.append(self.dataset[i])
        ret = []
        for i in range(len(data[0])):
            ret.append(
                Tensor(
                    np.stack([x[i] for x in data])
                )
            )
        return ret


def parse_mnist(image_filesname, label_filename):
    with gzip.open(image_filesname) as fid:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        X = np.frombuffer(fid.read(), 'B', offset=16)
        X = X.reshape(-1, 784).astype('float32') / 255
    with gzip.open(label_filename) as f:
        # First 8 bytes are magic_number, n_labels
        y = np.frombuffer(f.read(), 'B', offset=8)
    return X, y

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        X, y = parse_mnist(image_filename, label_filename)
        self.X = X
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        X_i = self.X[index]
        one_index = (X_i.ndim == 1)

        X_i = X_i.reshape(-1, 784)
        y_i = self.y[index].reshape(-1,)
        for j in range(X_i.shape[0]):
            X_ij = X_i[j].reshape(28, 28, 1)
            X_ij = self.apply_transforms(X_ij)
            X_i[j] = X_ij.reshape(-1)
        if one_index:
            X_i = X_i[0]
            y_i = y_i[0]
        return X_i, y_i

    def __len__(self) -> int:
        return len(self.X)

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
