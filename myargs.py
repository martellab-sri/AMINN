import argparse

parser = argparse.ArgumentParser(description='Train an AMINN model')

# ------------------------ Model related parameters ------------------------ #
parser.add_argument('--pooling', default='ave', type=str,
                    help='mode of MIL pooling')

parser.add_argument('--decay', default=0.005, type=float,
                    help='weight decay')

parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')

parser.add_argument('--recon_coef', default=1.0, type=float,
                    help='weight for reconstruction loss')

parser.add_argument('--fp_coef', default=1.0, type=float,
                    help='weight for bag precition loss')
# ------------------------ Training parameters ------------------------ #
parser.add_argument('--epoch', default=100, type=int,
                    help='number of epoch to train')

parser.add_argument('--runs', default=1, type=int,
                    help='number of repeated runs')

parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')

parser.add_argument('--folds', default=3, type=int,
                    help='number of folds for cross validation')

parser.add_argument('--do', default=0.0, type=float,
                    help='drop out rate')
# ------------------------ System parameters ------------------------ #
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='which gpu to use')

parser.add_argument('--out_csv',
                    default=r'./output.csv',
                    type=str, help='output file that stores hyper-parameters and performance')

args = parser.parse_args()
