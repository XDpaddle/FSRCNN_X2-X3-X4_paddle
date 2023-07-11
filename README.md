# FSRCNN_paddle
This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

参考：https://github.com/yjn870/FSRCNN-pytorch

## Requirements

- paddlepaddle 2.4.0
- paddleseg    2.8.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

- ## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset  | Scale | Type  | Link                                                         |
| -------- | ----- | ----- | ------------------------------------------------------------ |
| 91-image | 2     | Train | [Download](https://www.dropbox.com/s/01z95js39kgw1qv/91-image_x2.h5?dl=0) |
| 91-image | 3     | Train | [Download](https://www.dropbox.com/s/qx4swlt2j7u4twr/91-image_x3.h5?dl=0) |
| 91-image | 4     | Train | [Download](https://www.dropbox.com/s/vobvi2nlymtvezb/91-image_x4.h5?dl=0) |
| Set5     | 2     | Eval  | [Download](https://www.dropbox.com/s/4kzqmtqzzo29l1x/Set5_x2.h5?dl=0) |
| Set5     | 3     | Eval  | [Download](https://www.dropbox.com/s/kyhbhyc5a0qcgnp/Set5_x3.h5?dl=0) |
| Set5     | 4     | Eval  | [Download](https://www.dropbox.com/s/ihtv1acd48cof14/Set5_x4.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

```bash
python train.py --train-file "/root/autodl-tmp/paddle-FSRCNN/SR/BLAH_BLAH/91-image_x4.h5" \
                --eval-file "/root/autodl-tmp/paddle-FSRCNN/SR/BLAH_BLAH/Set5_x4.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 4 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 20 \
                --num-workers 0 \
                --seed 123                



- ## Test
权重文件位置：BLAH_BLAH/outputs

​```bash
python test.py --weights-file "/root/autodl-tmp/paddle-FSRCNN/SR/BLAH_BLAH/outputs/x3/best.pdiparams" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```

## Results

PSNR was calculated on the Y channel.

### Set5

| Eval. Mat | Scale | Paper | Ours (butterfly_GT.bmp) |
|-----------|-------|-------|-----------------|

| PSNR      | 3     | 28.68 | 28.00 |

