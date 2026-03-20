# -- coding: utf-8 --
import argparse

import torch
import torchvision
from thop import profile

from lib.networks import EMCADNet
parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
# network related parameters
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true',
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true',
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true',
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()
# Model
print('==> Building model..')
model = EMCADNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes, expansion_factor=args.expansion_factor, dw_parallel=not args.no_dw_parallel, add=not args.concatenation, lgag_ks=args.lgag_ks, activation=args.activation_mscb, encoder=args.encoder, pretrain= not args.no_pretrain).cuda() # model initialization for TransCASCADE
model = model.cuda()

dummy_input = torch.randn(1, 3, 224, 224).cuda()
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


