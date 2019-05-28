import os
import sys
import dill
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchtext import data
from torchtext.vocab import GloVe
from sklearn.metrics import classification_report

""" neural network """


class CNNText(nn.Module):

    def __init__(self, args):
        super(CNNText, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):

        x = self.embed(x)  # (N, W, D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


""" dataset loader """


def load_dataset(batch_size, path_data):
    tokenize = lambda x: x.split()
    captions = data.Field(sequential=True, tokenize=tokenize, lower=True,
                          include_lengths=True, batch_first=True, fix_length=1500)
    category = data.Field(sequential=False)
    channel_id = data.RawField()
    channel_id.is_target = False
    chunk = data.RawField()
    chunk.is_target = False
    video_id = data.RawField()
    video_id.is_target = False

    seeds_captions = data.TabularDataset(
        path=path_data, format='csv',
        skip_header=True,
        fields=[
            ('captions', captions),
            ('category', category),
            ('channel_id', channel_id),
            ('chunk', chunk),
            ('video_id', video_id)
        ]
    )

    train_data, test_data = seeds_captions.split(split_ratio=0.8, stratified=True, strata_field='category')
    train_data, valid_data = train_data.split(split_ratio=0.8, stratified=True, strata_field='category')

    captions.build_vocab(train_data, test_data, valid_data, vectors=GloVe(name='6B', dim=300))
    category.build_vocab(train_data, specials_first=False)

    vocab_size = len(captions.vocab)
    class_size = len(category.vocab) - 1

    print(vocab_size)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                   batch_size=batch_size,
                                                                   sort_key=lambda x: len(x.captions),
                                                                   repeat=False, shuffle=True)

    return captions, category, train_iter, valid_iter, test_iter, vocab_size, class_size


""" neural network util functions"""


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            # print(batch)
            (feature, size), target = batch.captions, batch.category
            if args.cuda:
                feature, size, target = feature.cuda(), size.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            # print('logit vector', logit.size(), logit)
            # print('target vector', target.size(), target)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            # print(loss)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss,
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval_test(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(args.captions, args.category, args.embed_num, args.class_num, model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(args.captions, args.category, args.embed_num, args.class_num, model, args.save_dir, 'snap', steps)


def eval_test(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0

    predicted_dict = {
        "true_cat": [],
        "pred_cat": [],
        "chunk": [],
        "video_id": [],
        "channel_id": []
    }
    for batch in data_iter:

        (feature, size), target = batch.captions, batch.category
        if args.cuda:
            feature, size, target = feature.cuda(), size.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

        predicted_dict["true_cat"] += list(batch.category.numpy())
        predicted_dict["pred_cat"] += list(torch.max(logit, 1)[1].view(target.size()).data.cpu().numpy())
        predicted_dict["chunk"] += batch.chunk
        predicted_dict["video_id"] += batch.video_id
        predicted_dict["channel_id"] += batch.channel_id

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

    df = pd.DataFrame(predicted_dict).groupby("channel_id").agg(lambda x: x.value_counts().index[0])

    print(classification_report(y_true=df["true_cat"].values, y_pred=df["pred_cat"].values))

    return accuracy


def save(captions, category, embed_num, class_num, model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save((captions, category, embed_num, class_num, model), save_path, pickle_module=dill)
