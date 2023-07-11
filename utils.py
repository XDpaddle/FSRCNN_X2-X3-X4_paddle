import paddle
import numpy as np


def calc_patch_size(func):

    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper

def calc_psnr(img1, img2):
    return 10. * paddle.log10(1. / paddle.mean((img1 - img2) ** 2))


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 
            25.064 * img[..., 2]) / 256.0
    else:
        return 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]
            ) / 256.0


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 *
            img[..., 2]) / 256.0
        cb = 128.0 + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 
            112.439 * img[..., 2]) / 256.0
        cr = 128.0 + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 *
            img[..., 2]) / 256.0
    else:
        y = 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]
            ) / 256.0
        cb = 128.0 + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]
            ) / 256.0
        cr = 128.0 + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]
            ) / 256.0
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256.0 + 408.583 * img[..., 2
            ] / 256.0 - 222.921
        g = 298.082 * img[..., 0] / 256.0 - 100.291 * img[..., 1
            ] / 256.0 - 208.12 * img[..., 2] / 256.0 + 135.576
        b = 298.082 * img[..., 0] / 256.0 + 516.412 * img[..., 1
            ] / 256.0 - 276.836
    else:
        r = 298.082 * img[0] / 256.0 + 408.583 * img[2] / 256.0 - 222.921
        g = 298.082 * img[0] / 256.0 - 100.291 * img[1] / 256.0 - 208.12 * img[
            2] / 256.0 + 135.576
        b = 298.082 * img[0] / 256.0 + 516.412 * img[1] / 256.0 - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.0
    x = paddle.to_tensor(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
