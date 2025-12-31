import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os


class OptInit:
    def __init__(self, logger):
        self.logger = logger
        parser = argparse.ArgumentParser(description='PyTorch implementation of STGCN for water quality prediction')

        # base
        parser.add_argument('--use_cpu', default=False, help='use cpu?')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--water_para_class', type=str, default='multi',
                            help='multi_var/single_var input')
        parser.add_argument('--water_level', type=str, default='multi',
                            help='multi/up/middle/down water level prediction')
        parser.add_argument('--pred_para_class', type=str, default='DO',
                            help='multi_var/single_var prediction')
        parser.add_argument('--input_step', type=int, default=48, help='input step')
        parser.add_argument('--pred_step', type=int, default=48, help='prediction step')
        parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default:64)')

        # train args
        parser.add_argument('--total_epochs', default=1000, type=int, help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2025, help='random seed')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='L2 weight decay')
        parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.75, type=float, help='learning rate decay')
        parser.add_argument('--optim_patience', default=25, type=int, help='learning rate decay patience epoch')

        # model args
        parser.add_argument('--n_filters', default=32, type=int, help='number of channels of deep features')
        parser.add_argument('--spatial_filters', default=16, type=int, help='number of spatial channels')

        parser.add_argument('--train_ratio', default=0.7, type=float, help='training data ratio') #0.3/0.5/0.7

        args = parser.parse_args()
        if args.water_level == 'multi':
            level_number = 3
        else:
            level_number = 1
        if args.water_para_class == 'multi':
            para_num = 5
        else:
            para_num = 1
        if args.pred_para_class == 'multi':
            predict_para_num = 5
        else:
            predict_para_num = 1
        if args.water_level == 'multi' and args.pred_para_class == 'multi':
            args.para_name = ['pH_U', 'turbidity_U', 'salinity_U', 'DO_U', 'temperature_U',
                   'pH_M', 'turbidity_M', 'salinity_M', 'DO_M', 'temperature_M',
                   'pH_D', 'turbidity_D', 'salinity_D', 'DO_D', 'temperature_D']
        elif args.water_level == 'up' and args.pred_para_class == 'multi':
            args.para_name = ['pH_U', 'turbidity_U', 'salinity_U', 'DO_U', 'temperature_U']
            args.adj_mat = np.load(os.path.join(args.data_dir, args.water_level+'_level_var_adj.npy'))
        elif args.water_level == 'middle' and args.pred_para_class == 'multi':
            args.para_name = ['pH_M', 'turbidity_M', 'salinity_M', 'DO_M', 'temperature_M']
        elif args.water_level == 'down' and args.pred_para_class == 'multi':
            args.para_name = ['pH_D', 'turbidity_D', 'salinity_D', 'DO_D', 'temperature_D']
        elif args.water_level == 'multi' and args.pred_para_class == 'DO':
            args.para_name = ['DO_U', 'DO_M', 'DO_D']
        args.adj_mat = np.load(os.path.join(args.data_dir, args.water_level+'_level_var_adj_{}.npy'.format(args.train_ratio)))
        args.water_para_num = level_number*para_num
        args.pred_para_num = level_number*predict_para_num
        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        args.task = args.pred_para_class+'_'+args.water_level+'_level_'+str(args.input_step)+'-'+str(args.pred_step)+'_step'+'_train'+str(args.train_ratio)
        args.save_dir = os.path.join('./save_model/STGCN_MRFA_block1/', args.task)
        self.args = args
        self._set_seed(self.args.seed)

        self.args.writer = SummaryWriter(log_dir=self.args.save_dir + '/log/', comment='comment',
                                         filename_suffix="_test_MRFA")
        # loss
        self.args.epoch = 0
        self.args.step = -1

        # self._configure_logger()
        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.logger.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            self.logger.info("{}:{}".format(arg, content))
        self.logger.info("==========     args END    =============")
        self.logger.info("\n")



    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



