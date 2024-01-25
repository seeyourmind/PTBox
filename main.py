# -*- coding: utf-8 -*-
"""
@author: 123456
"""
# Some libraries to import
import argparse, os, yaml
from Train import train

parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    default = 'LinkPrediction',
    type = str,
    help='choose different tasks')

parser.add_argument(
    '--model',
    default='PTBox',type=str,
    help='choose models')

parser.add_argument(
    '--dataset',
    default='yago',
    type=str, help='dataset to train on')

parser.add_argument(
    '--max_epoch',
    default=50000, type=int,
    help='number of total epochs (min value: 500)')

parser.add_argument('--dim', default=64, type=int)
parser.add_argument('--tk', default=20, type=int)
parser.add_argument('--mivmin', default=1e-2, type=float) # 1e-2
parser.add_argument('--mivmax', default=1.0, type=float)
parser.add_argument('--divmin', default=-0.1, type=float) # -0.1
parser.add_argument('--divmax', default=-1e-3, type=float) # -1e-3
parser.add_argument('--gbeta', default=0.001, type=float)

parser.add_argument(
    '--batch',
    default=5000, type=int,
    help='number of batch size')

parser.add_argument(
    '--lr',
    default=0.0001, type=float,
    help='number of learning rate')

parser.add_argument(
    '--gamma',
    default=1, type=float,
    help='number of margin')

parser.add_argument(
    '--loss_alpha',
    default=10, type=float,
    help='number of margin')

parser.add_argument(
    '--eta',
    default=100, type=int,
    help='number of negative samples per positive')

parser.add_argument(
    '--timedisc',
    default=1, type=int,
    help='method of discretizing time intervals')

parser.add_argument(
    '--cuda',
    default=True, type=bool,
    help='use cuda or cpu')

parser.add_argument(
    '--loss',
    default='loss', type=str,
    help='loss function, logloss')

parser.add_argument(
    '--cmin',
    default=0.005, type=float,
    help='cmin')

parser.add_argument(
    '--gran',
    default=1, type=int,
    help='time unit for ICEWS datasets')

parser.add_argument(
    '--thre',
    default=300, type=int,
    help='the mini threshold of time classes in yago and wikidata')

parser.add_argument(
    '--gpuID',
    default=0, type=int,
    help='the mini threshold of time classes in yago and wikidata')

parser.add_argument(
    '--comm',
    default='', type=str,
    help='remark of training script')

parser.add_argument('--revset', default=0, type=int)

parser.add_argument('-c', '--config', action='store_true', default=False, help='Use default config')

def load_config(args):
    pass

def main(args):
    if args.config:
        load_config(args)
    print(f'备注：{args.comm}')
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuID}'
    print(args)
    train(args,task=args.task,
          modelname=args.model,
          data_dir=args.dataset,
          dim=args.dim,
          batch=args.batch,
          lr =args.lr,
          max_epoch=args.max_epoch,
          gamma = args.gamma,
          lossname = args.loss,
          negsample_num=args.eta,
          timedisc = args.timedisc,
          cuda_able = args.cuda,
          cmin = args.cmin,
          gran = args.gran,
          count = args.thre,
          rev_set=args.revset,
          )              


if __name__ == '__main__':
    main(parser.parse_args())
