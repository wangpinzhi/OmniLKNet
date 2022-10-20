#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch.utils.data as data

from datasets.deep360 import Deep360Dataset


def build_data_loader(args):
    '''
    Build data loader

    :param args: arg parser object
    :return: train, validation and test dataloaders
    '''
    if args.dataset_directory == '':
        raise ValueError(f'Dataset directory cannot be empty.')
    else:
        dataset_dir = args.dataset_directory

    if args.dataset == 'deep360':
        dataset_train = Deep360Dataset(dataset_dir, 'train')
        dataset_validation = Deep360Dataset(dataset_dir, args.validation)
        dataset_test = Deep360Dataset(dataset_dir, 'test')

    else:
        raise ValueError(f'Dataset not recognized: {args.dataset}')

    data_loader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
    data_loader_validation = data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)

    print('train_dataset_len:',len(dataset_train))
    print('validation_dataset_len:',len(dataset_validation))
    print('test_dataset_len:',len(dataset_test))

    return data_loader_train, data_loader_validation, data_loader_test
