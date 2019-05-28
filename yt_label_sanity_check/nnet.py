from yt_label_sanity_check.utils import CNNText, load_dataset, train, eval_test
import pandas as pd
import numpy as np
import datetime
import argparse
import random
import torch
import dill
import os

parser = argparse.ArgumentParser(description='CNN `bread-n-butter` text classifier.')

""" Learning """

parser.add_argument('-path_data', type=str, default="./data/captions_seedsTrue_p100.csv",
                    help='Data for training [default: ./data/captions_seedsTrue_p100.csv]')


parser.add_argument('-lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')

parser.add_argument('-epochs', type=int, default=256,
                    help='number of epochs for train [default: 256]')

parser.add_argument('-batch-size', type=int, default=64,
                    help='batch size for training [default: 64]')

parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')

parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')

parser.add_argument('-save-interval', type=int, default=500,
                    help='how many steps to wait before saving [default:500]')

parser.add_argument('-save-dir', type=str, default='snapshot',
                    help='where to save the snapshot')

parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')

parser.add_argument('-save-best', type=bool, default=True,
                    help='whether to save when get best performance')

parser.add_argument('-shuffle', action='store_true', default=False,
                    help='shuffle the data every epoch')

""" Model """

parser.add_argument('-dropout', type=float, default=0.3,
                    help='the probability for dropout [default: 0.5]')

parser.add_argument('-max-norm', type=float, default=1.0,
                    help='l2 constraint of parameters [default: 3.0]')

parser.add_argument('-embed-dim', type=int, default=300,
                    help='number of embedding dimension [default: 128]')

parser.add_argument('-kernel-num', type=int, default=100,
                    help='number of each kind of kernel')

parser.add_argument('-kernel-sizes', type=str, default='3,4,5,6',
                    help='comma-separated kernel size to use for convolution')

parser.add_argument('-static', action='store_true', default=False,
                    help='fix the embedding')

""" Model """

parser.add_argument('-device', type=int, default=-1,
                    help='device to use for iterate data, -1 mean cpu [default: -1]')

parser.add_argument('-no-cuda', action='store_true', default=False,
                    help='disable the gpu')

""" Options """

parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')

parser.add_argument('-predict', type=str, default=None,
                    help='dest to a .csv file with "target" line')

parser.add_argument('-pred_dest', type=str, default="./data/predictions.csv",
                    help='dest to a file with a sentence per line')

parser.add_argument('-test', action='store_true', default=False,
                    help='train or test')

args = parser.parse_args()

""" Seeds """

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# Updates arguments
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

""" Actual training loop """

if args.predict is None:

    # Loads data
    args.captions, args.category, train_iter, dev_iter, test_iter, args.embed_num, args.class_num = \
        load_dataset(args.batch_size, args.path_data)

    # Loads model
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        _, _, _, _, cnn = torch.load(args.snapshot, pickle_module=dill)
    else:
        cnn = CNNText(args)

    # Sets cuda
    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    # Prints parameter
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.test:
        try:
            eval_test(test_iter, cnn, args)
        except Exception as e:
            print(e)
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print()
        try:
            train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')

else:
    import torch.autograd as autograd

    print('\nLoading model from {}...'.format(args.snapshot))
    args.captions, args.category, args.embed_num, args.class_num, cnn = torch.load(args.snapshot, pickle_module=dill)

    predictions = []
    chunksize = 1

    for chunk in pd.read_csv(args.predict, chunksize=chunksize):
        text = [[args.captions.vocab.stoi[x] for x in word.split()] for word in chunk.captions.values]
        x = torch.tensor([text[0]])
        x = autograd.Variable(x)
        if args.cuda:
            x = x.cuda()
        logit = cnn(x)
        predicted = torch.max(logit, 1)[1].view(1).data
        predictions += [args.category.vocab.itos[v] for v in predicted.cpu().numpy()]

    pd.DataFrame({"predictions": predictions}).to_csv(args.pred_dest, index=False)
