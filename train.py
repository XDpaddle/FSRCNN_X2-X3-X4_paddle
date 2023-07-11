from x2paddle import torch2paddle
import argparse
import os
import copy
import paddle
from paddle import nn
import paddle.optimizer as optim
from x2paddle.torch2paddle import DataLoader
from tqdm import tqdm
from models import FSRCNN
from datasets import TrainDataset
from datasets import EvalDataset
from utils import AverageMeter
from utils import calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--U', type=int, default=9)
    parser.add_argument('--num-features', type=int, default=128)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))  
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
    device=paddle.CUDAPlace(0)
    paddle.seed(args.seed)

    model = FSRCNN(scale_factor=args.scale)
 
    # for name, parameter in model.named_parameters():
    #     print('%-40s%-20s%s' %(name, parameter.requires_grad, parameter.is_leaf))

    criterion = paddle.nn.MSELoss() 
    
    optimizer = paddle.optimizer.Adam(parameters=[
                                                    {'params': model.first_part.parameters()},
                                                    {'params': model.mid_part.parameters()},
                                                    {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
                                                ], learning_rate=args.lr)
    # optimizer = paddle.optimizer.Adam([{'parameters': model.first_part.parameters()},
    #                                 {'parameters': model.mid_part.parameters()},
    #                                 {'parameters': model.last_part.parameters(), 'parameters': args.lr * 0.1}], parameters=args.lr)
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers, 
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len(train_dataset) - len(train_dataset) % args.batch_size, ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                preds = model(inputs)
                # print(paddle.shape(inputs),paddle.shape(labels),paddle.shape(preds))
                
                loss = criterion(preds, labels)

                epoch_losses.update(loss.numpy()[0], len(inputs))

                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        # print(inputs.shape)
        paddle.save(model.state_dict(), os.path.join(args.outputs_dir,'epoch_{}.pdiparams'.format(epoch)))
        
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with paddle.no_grad():
                preds = model(inputs).clip(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg.numpy()[0]))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

       

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr.numpy()[0]))
        paddle.save(best_weights, os.path.join(args.outputs_dir, 'best.pdiparams'))


        # with tqdm(total=len(train_dataset) - len(train_dataset) % args.batch_size, ncols=80) as t:
        #     t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            
        #     for data in train_dataloader:
        #         inputs, labels = data

        #         inputs = inputs.to(device)
        #         labels = labels.to(device)

        #         preds = model(inputs)

        #         loss = criterion(preds, labels)

        #         epoch_losses.update(loss.numpy()[0], len(inputs))

        #         optimizer.clear_grad()
        #         loss.backward()
        #         optimizer.step()

        #         t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg.numpy()[0]))
        #         t.update(len(inputs))
        # print(inputs.shape)
        # paddle.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pdiparams'.format(epoch)))
       
        # model.eval()
        # epoch_psnr = AverageMeter()

        # for data in eval_dataloader:
        #     inputs, labels = data

        #     inputs = inputs.to(device)
        #     labels = labels.to(device)

        #     with paddle.no_grad():
        #         preds = model(inputs).clip(0.0, 1.0)

        #     epoch_psnr.update(calc_psnr(preds, labels).numpy()[0], len(inputs))
        
        # print('eval psnr: {:.2f}'.format(epoch_psnr.avg.numpy()[0]))

        # if epoch_psnr.avg.numpy()[0] > best_psnr:
        #     best_epoch = epoch
        #     best_psnr = epoch_psnr.avg.numpy()[0]
        #     best_weights = copy.deepcopy(model.state_dict())

