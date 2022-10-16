import argparse
import collections
from data_loader.data_loaders import *
import models.loss as module_loss
import models.metric as module_metric
import models.Pearattention.GAT_Pearson as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # get function handles of loss and metrics
    criterion_1 = getattr(module_loss, config['loss_1'])
    criterion_2 = getattr(module_loss, config['loss_2'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer and trainer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], batch_size)
    weights_for_each_class = calc_class_weight(data_count)

    trainer = Trainer(model, criterion_1, criterion_2, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    trainer.training()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)
    config = ConfigParser.from_args(args, fold_id, options)
    folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    main(config, fold_id)

