import argparse
import paddle
import numpy as np
import PIL.Image as pil_image
from models import FSRCNN
from utils import convert_ycbcr_to_rgb
from utils import preprocess
from utils import calc_psnr
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    device = paddle.CUDAPlace(0)

    model = FSRCNN(scale_factor=args.scale)
    state_dict = model.state_dict()
    for n, p in paddle.load(args.weights_file).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
        
    model.eval()
    image = pil_image.open(args.image_file).convert('RGB')
    image_width = image.width // args.scale * args.scale
    image_height = image.height // args.scale * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale),resample=pil_image.BICUBIC)

    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale),resample=pil_image.BICUBIC)

    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with paddle.no_grad():
        preds = model(lr).clip(0.0, 1.0)
    psnr = calc_psnr(hr, preds)
    print(psnr)
    print('PSNR: {:.2f}'.format(psnr.numpy()[0]))
    # preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    preds = (preds* 255.0).numpy().squeeze().squeeze()
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_fsrcnn_x{}.'.format(args.scale)))