"""Produces word-index dictionary."""
"""TODO
vocabulary size limit
"""
import re
import random
import torch
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP

UNK_token = 0
class Preprocess:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: 'UNK'}
        self.word2count = {}
        self.n_words = 1
        self.tag2idx = {'ne': 0, 'hp': 1, 'sd': 2, 'ag': 3, 'dg': 4,
                        'sp': 5, 'fr': 6}

    def addSentence(self, tokenized_sentence):
        for word in tokenized_sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterText(text):
    text = text.lower().strip()
    text = re.sub(r"([.:;)(<>-_])", "", text)
    text = text.strip()
    return text

def blogsToDictionary(sentences, MAX_LENGTH = 30):
    nlp = StanfordCoreNLP('/Users/jaeickbae/Documents/projects/utils/stanford\
-corenlp-full-2017-06-09')
    dictionary = {'ne': {'tokens': [], 'sentences':[]},
                'hp': {'tokens': [], 'sentences':[]},
                'sd': {'tokens': [], 'sentences':[]},
                'ag': {'tokens': [], 'sentences':[]},
                'dg': {'tokens': [], 'sentences':[]},
                'sp': {'tokens': [], 'sentences':[]},
                'fr': {'tokens': [], 'sentences':[]}}
    min_length = MAX_LENGTH
    max_length = MAX_LENGTH
    longest_sentence = ''
    for s in sentences:
        words = s.split()
        sentence = ' '.join(words[2:])
        filtered_sentence = filterText(sentence)
        tokenized_sentence = nlp.word_tokenize(filtered_sentence)
        if len(tokenized_sentence) < 5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        dictionary[words[0]]['tokens'].append(tokenized_sentence)
        dictionary[words[0]]['sentences'].append(sentence)
        if len(tokenized_sentence) < min_length:
            min_length = len(tokenized_sentence)
        if len(tokenized_sentence) >= max_length:
            max_length = len(tokenized_sentence)
            longest_sentence = sentence

    #print stat
    print('**************Statistics**************')
    print('longest: ' + str(max_length))
    print('shortest: '+ str(min_length))
    print('Longest Sentence example: ' + longest_sentence)

    for key in dictionary:
        print(key + ':' + str(len(dictionary[key]['tokens'])))
    print('**************************************')
    return dictionary

def split_train_and_test_data(data, train = 0.8, test = 0.2):
    train_data = {}
    test_data = {}
    for key in data:
        zipped = list(zip(data[key]['tokens'], data[key]['sentences']))
        random.shuffle(zipped)
        data[key]['tokens'], data[key]['sentences'] = zip(*zipped)
        train_portion = int(len(data[key]['tokens']) * train)
        train_data[key] = {'tokens': [], 'sentences': []}
        test_data[key] = {'tokens': [], 'sentnces': []}
        train_data[key]['tokens'] = data[key]['tokens'][:train_portion]
        test_data[key]['tokens'] = data[key]['tokens'][train_portion+1:]
        train_data[key]['sentences'] = data[key]['sentences'][:train_portion]
        test_data[key]['sentences'] = data[key]['sentences'][train_portion+1:]
    return train_data, test_data

def get_pairs(train_data, test_data):
    train_pair = []
    test_pair = []
    test_sentence = []
    for key in train_data:
        for item in train_data[key]['tokens']:
            train_pair.append([item, key])
    random.shuffle(train_pair)

    for key in test_data:
        for idx, item in enumerate(test_data[key]['tokens']):
            test_pair.append([item, key])
            test_sentence.append(test_data[key]['sentences'])
    zipped = list(zip(test_pair, test_sentence))
    random.shuffle(zipped)
    test_pair, test_sentence = zip(*zipped)
    return train_pair, test_pair, test_sentence

def change_to_variables(train_pair, test_pair, CUDA_use):
    train_input = Preprocess('train')
    for pair in train_pair:
        train_input.addSentence(pair[0])

    train_input_idx = []
    for pair in train_pair:
        train_input_idx.append([train_input.word2index[word]
         for word in pair[0]])
    #train_input_var = [idx_list for idx_list in train_input_idx]

    if CUDA_use:
        train_input_var = [Variable(torch.LongTensor(idx_list).view(-1,1)).cuda()
        for idx_list in train_input_idx]
    else:
        train_input_var = [Variable(torch.LongTensor(idx_list).view(-1,1))
        for idx_list in train_input_idx]

    train_output_label = [[train_input.tag2idx[pair[1]]] for pair in train_pair]

    test_input_idx = []
    for pair in test_pair:
        sequence = []
        for word in pair[0]:
            if word in train_input.word2index:
                sequence.append(train_input.word2index[word])
            else:
                sequence.append(UNK_token)
        test_input_idx.append(sequence)
    #test_input_var = [idx_list for idx_list in test_input_idx]

    if CUDA_use:
        test_input_var = [Variable(torch.LongTensor(idx_list).view(-1,1)).cuda()
        for idx_list in test_input_idx]
    else:
        test_input_var = [Variable(torch.LongTensor(idx_list).view(-1,1))
        for idx_list in test_input_idx]

    test_output_label = [[train_input.tag2idx[pair[1]]] for pair in test_pair]

    print('train words ' + str(train_input.n_words))
    return train_input_var, train_output_label,\
    test_input_var, test_output_label, train_input

def prepareData(sentence_list, CUDA_use = False, MAX_LENGTH=30, MAX_VOCAB=80000):
    # make dictionary
    # split into train and test data
    # make into pairs (train_pair, test_pair)
    # make pairs into variables two pairs of
    # (train_input_vec, train_ouput_label)
    # return train_input_var, train_ouptut_label, test_input_var,
    # test_output_label, test_sentence
    print("Making into Dictionary...")
    dictionary = blogsToDictionary(sentence_list, MAX_LENGTH)
    print("Chaning to Variables...")
    train_data, test_data = split_train_and_test_data(dictionary)
    train_pair, test_pair, test_sentence = get_pairs(train_data, test_data)
    train_input_var, train_output_label, test_input_var, test_output_label,\
    train_input = change_to_variables(train_pair, test_pair, CUDA_use)
    print("Data Preparation Done.")
    return train_input_var, train_output_label, test_input_var,\
    test_output_label, test_sentence, train_input

"""
if __name__ == "__main__":
    data = '/Users/jaeickbae/Documents/projects/2017 Affective Computing\
/Emotion-Data/Benchmark/category_gold_std.txt'
    with open(data, 'r') as f:
        sentences = f.readlines()
    prepareData(sentences)
"""
