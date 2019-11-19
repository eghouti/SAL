import argparse
import math

binarized = False
display = False
temperature = 0.3
maxvalue = 0
heat_map= False
distributed=False
nb_params = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--feature_maps', default=83, type=int, help='Number of feature_maps for first blocks')
parser.add_argument('--kernel_size', default=3, type=int, help='Kernel size')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--dropout', default=0., type=float, help='dropout')
parser.add_argument('--temperature_init', default=20, type=float, help='initial temperature')
parser.add_argument('--temperature_final', default=70, type=float, help='final temperature')
parser.add_argument('--epochs', default=300, type=int, help='epochs per era')
parser.add_argument('--dataset', default="cifar10", help='dataset to use')
#parser.add_argument('--arch', default="resnet18", help='arch to use, can be resnet18 or resnet20')
parser.add_argument('--shared', action="store_true", help='share shift accross feature maps')

args = parser.parse_args()
temperature = args.temperature_init
temperature_final=args.temperature_final
temperature_update = math.exp(math.log(args.temperature_final / args.temperature_init) / (391 * args.epochs))# * 3))

