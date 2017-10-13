"""GRNN implementation for fine-grained emotion classification."""
"""We use blogs data instead of twitter data annotated and used in the paper.
Furthermore, I used Ekman's emotion set since blogs data is annotated with
Ekman's emotions instead of emotions defined by Plutchik."""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from preprocess import *
from utils import *


class GRNN(nn.Module):
    def __init__(self, input_size, embedding_size, batch_size, hidden_size,
     label_size, MAX_LENGTH, n_layer=1, CUDA_use=False):
        super(GRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.MAX_LENGTH = MAX_LENGTH
        self.n_layer = n_layer
        self.CUDA_use = CUDA_use

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(hidden_size, label_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, batch_word_seq, hidden):
        # batch_word_seq is a variable
        batch_size = batch_word_seq.size()[0]
        seq_len = batch_word_seq.size()[1]
        embedded = self.embedding(batch_word_seq).view(seq_len, batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)[-1].view(batch_size, seq_len, -1)
        #seq_len, batch, hidden_size -> batch, hidden_size
        output = self.softmax(self.out(output))
        #batch, label_size
        return output

    def init_hidden(self, batch_size):
        if self.CUDA_use:
            hidden = Variable(torch.zeros(self.n_layer, batch_size,
            self.hidden_size)).cuda()
        else:
            hidden = Variable(torch.zeros(self.n_layer, batch_size,
            self.hidden_size))

def train(grnn, grnn_optimizer, criterion, input_variables, labels):
    #input_variables = seq_len * batch
    batch_size = input_variables.size()[1]
    grnn_init_hidden = grnn.init_hidden(batch_size)
    grnn_optimizer.zero_grad()
    output = grnn(input_variables, grnn_init_hidden)
    # output shape = batch, label_size
    loss = 0
    for oi in range(batch_size):
        topv, topi = output[oi].data.topk(1)
        predicted = topi[0][0]
        loss += criterion(output[oi], labels[oi])
    loss.backward()
    grnn_optimizer.step()
    return loss.data[0]/(batch_size*1.)

def test(grnn, test_variables, test_labels):
    batch_size = test_variables.size()[1]
    grnn_init_hidden = grnn.init_hidden(batch_size)
    output, hidden = grnn(test_variables, grnn_init_hidden)
    acc = 0
    predict_results = []
    for oi in range(batch_size):
        topv, topi = output[oi].data.topk(1)
        predicted = topi[0][0]
        predict_results.append(predicted)
        if predicted == test_labels[oi]:
            acc += 1
    return acc / batch_size*1., predict_results

if __name__ == "__main__":
    CUDA_use = False
    UNK_token = 0
    n_epoch = 7
    n_iter = 100
    n_layer = 3
    embedding_size = 300
    hidden_size = 1000
    label_size = 7
    mini_batch_size = 1
    learning_rate = 0.0001
    MAX_LENGTH = 30
    data = '/Users/jaeickbae/Documents/projects/2017 Affective Computing\
/Emotion-Data/Benchmark/category_gold_std.txt'

    # get data
    with open(data, 'r') as f:
        sentences = [line for line in f.readlines()]
    train_input_var, train_output_label, test_input_var,\
    test_output_label, test_sentence, train_input = \
    prepareData(sentences, CUDA_use, MAX_LENGTH)
    if CUDA_use:
        train_output_label = Variable(torch.LongTensor(train_output_label)).cuda()
        test_output_label = Variable(torch.LongTensor(test_output_label)).cuda()
    else:
        train_output_label = Variable(torch.LongTensor(train_output_label))
        test_output_label = Variable(torch.LongTensor(test_output_label))

    """
    train_input_var = torch.LongTensor(train_input_var)
    test_input_var = torch.LongTensor(test_input_var)
    train_output_label = torch.LongTensor(train_output_label)
    test_output_label = torch.LongTensor(test_output_label)
    train = data_utils.TensorDataset(train_input_var, train_output_label)
    test = data_utils.TensorDataset(test_input_var.data, test_output_label)
    train_loader = data_utils.DataLoader(train, batch_size=mini_batch_size)
    test_loader = data_utils.DataLoader(test, batch_size=mini_batch_size)
    """

    # define model, criterion, and optimizer
    if CUDA_use:
        grnn = GRNN(train_input.n_words, embedding_size, mini_batch_size,\
        hidden_size, label_size, MAX_LENGTH, n_layer, CUDA_use).cuda()
    else:
        grnn = GRNN(train_input.n_words, embedding_size, mini_batch_size,\
        hidden_size, label_size, MAX_LENGTH, n_layer, CUDA_use)
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
            """
            if CUDA_use:
                sentences = Variable(torch.LongTensor(sentences)).view\
                (mini_batch_size,-1,1).cuda()
            else:
                sentences = Variable(torch.LongTensor(sentences)).view\
                (mini_batch_size,-1,1)
            """
            loss = train(grnn, grnn_optimizer, criterion, sentences, \
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
    showPlot(plot_losses)
    # test
    # save model
    torch.save(grnn.save_dict(), './grnn_model.pkl')