import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set Parameters for the Task', add_help=False)
    
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float) # init learning rate
    parser.add_argument('--lr_decay_rate', default=0.99, type=float) 
    parser.add_argument('--batch_size', default=8, type=int) # batch size
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs',default=100, type=int) # epochs
    parser.add_argument('--device',default='cuda',type=str) # device for model inference
    parser.add_argument('--seed', default=42, type=int) # training seed
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--train', action='store_true', default=False) # True is for train, false if for test
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')

    # Dataset parameters
    parser.add_argument('--dataset', default='deep360', type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='', type=str, help='directory to dataset')

    # data settings
    parser.add_argument('--input_size', default=(3,1024,512), help='input data size: CxHxW')
    parser.add_argument('--maxdisp', default=511)

    parser.add_argument('--start_epoch', default=1, type=int)
    
    return parser