import h5py
import numpy as np
from paddle.io import Dataset
import paddle


class TrainDataset(paddle.io.Dataset):

    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
           with h5py.File(self.h5_file, 'r') as f:
            out1,out2=np.expand_dims(f['lr'][idx] / 255.0, 0), np.expand_dims(f['hr'][idx] / 255.0, 0)

            if isinstance(out1, paddle.Tensor):
                out1 = out1.numpy()
            if isinstance(out2, paddle.Tensor):
                out2 = out2.numpy()
            return out1,out2
            # return np.expand_dims(f['lr'][idx] / 255.0, 0), np.expand_dims(f['hr'][idx] / 255.0, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(paddle.io.Dataset):

    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):

        with h5py.File(self.h5_file, 'r') as f:
            out1,out2=np.expand_dims(f['lr'][str(idx)][:, :] / 255.0, 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255.0, 0)
            if isinstance(out2, paddle.Tensor):
                out1=out1.numpy()
            if isinstance(out2, paddle.Tensor):
                out2 = out2.numpy()
            return out1,out2
            # return np.expand_dims(f['lr'][str(idx)][:, :] / 255.0, 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255.0, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

