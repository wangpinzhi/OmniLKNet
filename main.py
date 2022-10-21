import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm


from models.omnilknet import OmniLKNet 
from utilities import Saver, TensorboardSummary, get_args_parser, Metric
from datasets import build_data_loader

def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)

def write_summary(stats:dict, summary:TensorboardSummary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    for key in stats.keys():
        summary.writer.add_scalar(f'{mode}\{key}', stats[key], epoch)

def print_param(model):
    """
    print number of parameters in the model
    """
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'first_step' in n and p.requires_grad)
    print('number of params in first_step:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'second_step' in n and p.requires_grad)
    print('number of params in second_step:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'final_step' in n and p.requires_grad)
    print('number of params in final_step:', f'{n_parameters:,}')

def train(model, optimizer, lr_scheduler, prev_best, checkpoint_saver, device,  summary, args, data_loader_train, data_loader_val):
    for epoch in range(args.start_epoch, args.epochs+1):
        loss_for_one_epoch = 0.0
        epoch_metric = Metric()
        
        '''
        Train for one epoch
        '''
        with tqdm(total=len(data_loader_train)) as t:
            for batch_idx, batchData in enumerate(data_loader_train):
                
                # Size: B C H W
                leftImg = batchData['left'].to(device)
                rightImg = batchData['right'].to(device)
                disp_gt = batchData['disp'].to(device)
                disp_mask = batchData['occ_mask'].to(device)
                disp_mask = ~disp_mask

                # model inference
                model.train()
                optimizer.zero_grad()
                output1, output2, output3 = model(leftImg, rightImg)

                # Loss --------------------------------------------
                loss = 0.5 * F.smooth_l1_loss(output1[disp_mask],
                                    disp_gt[disp_mask],
                                    size_average=True) + 0.7 * F.smooth_l1_loss(output2[disp_mask],
                                                                                disp_gt[disp_mask],
                                                                                size_average=True) + F.smooth_l1_loss(output3[disp_mask],
                                                                                                                    disp_gt[disp_mask],
                                                                                                                    size_average=True)
                # --------------------------------------------------
                cur_loss = loss.data.item()
                loss_for_one_epoch += cur_loss
                loss.backward()
                optimizer.step()

                epoch_metric.add(output1[disp_mask], disp_gt[disp_mask])
                epoch_metric.update()

                t.set_description(desc=f'Epoch[{epoch}/{args.epochs}]: ')
                t.set_postfix(loss=cur_loss, cur_epe=epoch_metric.cur_Epe,cur_e3px=epoch_metric.cur_E3px,cur_e5px=epoch_metric.cur_E5px)
                t.update(1)
                
        epoch_metric.average()
        stats = {
            'epe':epoch_metric.avg_Epe,
            '3px error':epoch_metric.avg_E3px,
            '5px error':epoch_metric.avg_E5px,
            'loss':loss_for_one_epoch / epoch_metric.counter
        }
        write_summary(stats, summary, epoch, 'training')  # tensorboardX for iter
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False)

        '''
        Validation for one epoch
        '''
        with torch.no_grad():
            torch.cuda.empty_cache()
            with tqdm(total=len(data_loader_val)) as t:
                for batch_idx, batchData in enumerate(data_loader_val):
                   
                    leftImg = batchData['left'].to(device)
                    rightImg = batchData['right'].to(device)
                    disp_gt = batchData['disp'].to(device)
                    disp_mask = batchData['occ_mask'].to(device)

                    

                    # model inference
                    model.eval()

                    output1, _, _ = model(leftImg, rightImg)

                    cur_loss = loss.data.item()
                    loss_for_one_epoch += cur_loss

                    epoch_metric.add(output1[disp_mask], disp_gt[disp_mask])
                    epoch_metric.update()

                    t.set_description(desc=f'Epoch[{epoch}/{args.epochs}]: ')
                    t.set_postfix(loss=cur_loss, cur_epe=epoch_metric.cur_Epe,cur_e3px=epoch_metric.cur_E3px,cur_e5px=epoch_metric.cur_E5px)
                    t.update(1) 

        # save if best
        epoch_metric.average()
        if prev_best > epoch_metric.avg_Epe and 0.5 > epoch_metric.avg_E3px :
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, True)



            




def main(args):

    # get device
    device = torch.device(args.device)
    if args.device =='cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark=True

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = OmniLKNet(args).to(device)
    print_param(model)

    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

     # load checkpoint if provided
    prev_best = np.inf
    if args.resume:
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)

        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        # check missing and unexpected keys
        if len(missing) > 0:
            print("Missing keys: ", ','.join(missing))
            raise Exception("Missing keys.")
        unexpected_filtered = [k for k in unexpected if
                               'running_mean' not in k and 'running_var' not in k]  # skip bn params
        if len(unexpected_filtered) > 0:
            print("Unexpected keys: ", ','.join(unexpected_filtered))
            raise Exception("Unexpected keys.")
        print("Pre-trained model successfully loaded.")

        # if not ft/inference/eval, load states for optimizer, lr_scheduler, amp and prev best
        if args.train:
            if len(unexpected) > 0:  # loaded checkpoint has bn parameters, legacy resume, skip loading
                raise Exception("Resuming legacy model with BN parameters. Not possible due to BN param change. " + 
                "Do you want to finetune or inference? If so, check your arguments.")
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                prev_best = checkpoint['best_pred']
                print("Pre-trained optimizer, lr scheduler and stats successfully loaded.")

    # initiate saver and logger
    checkpoint_saver = Saver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    # build dataloader
    data_loader_train, data_loader_val, data_loader_test = build_data_loader(args)

    # train
    if args.train:
        print('Start Training')
        train(model, optimizer, lr_scheduler, prev_best, checkpoint_saver, device,  summary_writer, args, data_loader_train, data_loader_val)
        print('End   Training')
    else:
        print('Start Testing')
        print('End   Testing')
    # test
    
    

if __name__ == '__main__':

   args = get_args_parser().parse_args()
   main(args)
    