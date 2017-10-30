"""GRNN implementation for fine-grained emotion classification."""
"""We use various other data instead of twitter data annotated and used in the
paper. Furthermore, Ekman's emotion set is used instead of emotion categories
 defined by Plutchik."""

import sys
import os
import time
import json
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from preprocess import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class GRNN(nn.Module):
    def __init__(self, input_size, embedding_size, mini_batch_size, hidden_size,
     label_size, MAX_LENGTH, n_layer=1, CUDA_use=False):
        super(GRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.MAX_LENGTH = MAX_LENGTH
        self.n_layer = n_layer
        self.CUDA_use = CUDA_use

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=0.1)
        self.out = nn.Linear(hidden_size, label_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, batch_word_seq, hidden):
        # batch_word_seq is a variable
        batch_size = batch_word_seq.size()[0]
        seq_len = batch_word_seq.size()[1]
        embedded = self.embedding(batch_word_seq).view(seq_len, batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        #seq_len, batch, hidden_size -> batch, hidden_size
        # use last output
        output = output[-1].view(batch_size, -1)
        output = self.out(output)
        #batch, label_size
        return output

    def init_hidden(self, mini_batch_size):
        if self.CUDA_use:
            hidden = Variable(torch.zeros(self.n_layer, mini_batch_size,
            self.hidden_size)).cuda()
        else:
            hidden = Variable(torch.zeros(self.n_layer, mini_batch_size,
            self.hidden_size))
        return hidden

def train(grnn, grnn_optimizer, criterion, input_variables, labels):
    #input_variables = batch(1) * seq_len
    batch_size = input_variables.size()[0]
    grnn_init_hidden = grnn.init_hidden(batch_size)
    grnn_optimizer.zero_grad()
    output = grnn(input_variables, grnn_init_hidden)
    # output shape = batch, label_size
    loss = criterion(output, labels)
    loss.backward()
    grnn_optimizer.step()
    return loss.data[0]/(batch_size*1.)

def test(grnn, test_variables, test_labels):
    grnn.train(False)
    batch_size = test_variables.size()[0] #1
    grnn_init_hidden = grnn.init_hidden(batch_size)
    output = grnn(test_variables, grnn_init_hidden)
    output = nn.LogSoftmax(output)
    acc = 0
    for oi in range(batch_size):
        topv, topi = output[oi].data.topk(1)
        predicted = topi[0]
        if predicted == test_labels.data[0]:
            acc += 1
    grnn.train(True)
    return acc / batch_size*1., predicted

def confusionMatrix(y_pred, y_true, fname):
    mat = confusion_matrix(y_true, y_pred)
    precision, recall, fscore, _ = \
    precision_recall_fscore_support(y_true, y_pred)
    with open(fname, 'w') as f:
        for row in mat:
            for num in row:
                f.write(str(num) + '\t')
            f.write('\n')
        f.write('\n')
        for idx, item in enumerate(precision):
            f.write(str(item)+'\t'+str(recall[idx])+'\t'+str(fscore[idx])+'\n')

if __name__ == "__main__":
    CUDA_use = True
    UNK_token = 0
    n_epoch = 14
    n_iter = 100
    n_layer = 3
    embedding_size = 300
    hidden_size = 1000
    label_size = 7
    mini_batch_size = 1
    learning_rate = 0.0001
    MAX_LENGTH = 30
    data_name = 'bopang'
    blogs_data = '/Users/jaeickbae/Documents/projects/2017 Affective Computing\
/Emotion-Data/Benchmark/category_gold_std.txt'
    bopang_data = '/Users/jaeickbae/Documents/projects/data/bopang_twitter/rt-polaritydata'

    # get data
    train_input_var, train_output_label, test_input_var,\
    test_output_label, test_sentence, train_input = \
    prepareData(data_name , bopang_data, CUDA_use, MAX_LENGTH)

    train_output_label = Variable(torch.LongTensor(train_output_label))
    test_output_label = Variable(torch.LongTensor(test_output_label))
    if CUDA_use:
        train_output_label = train_output_label.cuda()
        test_output_label = test_output_label.cuda()
    # define model, criterion, and optimizer
    grnn = GRNN(train_input.n_words, embedding_size, mini_batch_size,\
    hidden_size, label_size, MAX_LENGTH, n_layer, CUDA_use)
    if CUDA_use:
        grnn = grnn.cuda()
    criterion = nn.CrossEntropyLoss()
    grnn_optimizer = torch.optim.Adam(grnn.parameters(), lr=learning_rate)

    # train
    start = time.time()
    plot_losses = []
    plot_loss_total = 0
    print_loss_total = 0
    print_every = 1000
    plot_every = 100
    n_iter = 0
    for epoch in range(n_epoch):
        for i, sentences in enumerate(train_input_var):
            n_iter += 1
            loss = train(grnn, grnn_optimizer, criterion, sentences,\
            train_output_label[i])
            print_loss_total += loss
            plot_loss_total += loss

            if (n_iter) % print_every == 0:
                print_loss_avg = print_loss_total / (print_every*1.)
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % ((timeSince(start, n_iter/
                (len(train_input_var)*n_epoch*1.))), n_iter,
                n_iter/(len(train_input_var)*n_epoch*1.)*100, print_loss_avg))

            if n_iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    fname = data_name + str(MAX_LENGTH)
    showPlot(plot_losses, fname + '_loss.png')

    torch.save(grnn.state_dict(), fname + '_model.pkl')

    # test
    test_acc_total = 0
    y_pred = []
    for i in range(len(test_input_var)):
        acc, predicted = test(grnn, test_input_var[i], test_output_label[i])
        test_acc_total += acc
        y_pred.append(predicted)
    print('acc: ' + str(test_acc_total / (len(test_input_var)*1.)))
    y_true = [label.data[0] for label in test_output_label]
    confusionMatrix(y_pred, y_true, fname +'_confusion.txt')
