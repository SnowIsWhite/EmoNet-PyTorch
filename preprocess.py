"""Produces word-index dictionary."""
"""TODO
vocabulary size limit
"""
import re
import random
import torch
import json
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
        if name == 'blogs':
            self.tag2idx = {'ne': 0, 'hp': 1, 'sd': 2, 'ag': 3, 'dg': 4,
            'sp': 5, 'fr': 6}
        elif name == 'twitter':
            self.tag2idx = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'sadness': 4, 'surprise': 5}

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
    text = re.sub(r"([.:;)(<>-_!?\"\'])", r"\1", text)
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

    total = 0
    for key in dictionary:
        print(key + ':' + str(len(dictionary[key]['tokens'])))
        total += len(dictionary[key]['tokens'])
    print('total : '+str(total))
    print('**************************************')
    with open('./blogs'+str(MAX_LENGTH)+'.json', 'w') as jsonfile:
        json.dump(dictionary, jsonfile)
    return dictionary

def twitterToDictionary(sentences, MAX_LENGTH=30):
    # Twitter Emotion Corpus
    nlp = StanfordCoreNLP('/Users/jaeickbae/Documents/projects/utils/stanford\
-corenlp-full-2017-06-09')
    dictionary = {'anger': {'tokens': [], 'sentences':[]},
                'disgust': {'tokens': [], 'sentences':[]},
                'fear': {'tokens': [], 'sentences':[]},
                'joy': {'tokens': [], 'sentences':[]},
                'sadness': {'tokens': [], 'sentences':[]},
                'surprise': {'tokens': [], 'sentences':[]}}
    min_length = MAX_LENGTH
    max_length = MAX_LENGTH
    longest_sentence = ''
    for s in sentences:
        phrase = s.split('\t')
        # phrase[0]: twitter number
        # phrase[1]: content
        # phrase[2]: tag
        tag = phrase[2][3:].strip()
        if tag not in dictionary:
            print(tag)
            continue
        sentence = phrase[1]
        filtered_sentence = filterText(sentence)
        tokenized_sentence = nlp.word_tokenize(filtered_sentence)
        if len(tokenized_sentence) < 5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        dictionary[tag]['tokens'].append(tokenized_sentence)
        dictionary[tag]['sentences'].append(sentence)
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

    total = 0
    for key in dictionary:
        print(key + ':' + str(len(dictionary[key]['tokens'])))
        total += len(dictionary[key]['tokens'])
    print('total : '+str(total))
    print('**************************************')

    with open('./twitter'+str(MAX_LENGTH)+'.json', 'w') as jsonfile:
        json.dump(dictionary, jsonfile)
    return dictionary

def splitTrainAndTestData(data, train = 0.8, test = 0.2):
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

def getPairs(train_data, test_data):
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

def changeToVariables(train_pair, test_pair, CUDA_use, data_name, MAX_VOCAB):
    train_input = Preprocess(data_name)
    for pair in train_pair:
        train_input.addSentence(pair[0])

    # Vocabulary frequency check
    print('train words ' + str(train_input.n_words))
    if train_input.n_words > MAX_VOCAB:
        pass

    train_input_idx = []
    for pair in train_pair:
        train_input_idx.append([train_input.word2index[word]
         for word in pair[0]])
    #train_input_var = [idx_list for idx_list in train_input_idx]

    if CUDA_use:
        train_input_var = \
        [Variable(torch.LongTensor(idx_list).unsqueeze(0)).cuda()
        for idx_list in train_input_idx]
    else:
        train_input_var = [Variable(torch.LongTensor(idx_list).unsqueeze(0))
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
        test_input_var = \
        [Variable(torch.LongTensor(idx_list).unsqueeze(0)).cuda()
        for idx_list in test_input_idx]
    else:
        test_input_var = [Variable(torch.LongTensor(idx_list).unsqueeze(0))
        for idx_list in test_input_idx]

    test_output_label = [[train_input.tag2idx[pair[1]]] for pair in test_pair]

    return train_input_var, train_output_label,\
    test_input_var, test_output_label, train_input

def prepareData(sentence_list, data_name, data_dir, CUDA_use = False,
MAX_LENGTH=30, MAX_VOCAB=80000):
    # make dictionary
    # split into train and test data
    # make into pairs (train_pair, test_pair)
    # make pairs into variables two pairs of
    # (train_input_vec, train_ouput_label)
    # return train_input_var, train_ouptut_label, test_input_var,
    # test_output_label, test_sentence
    print("Making into Dictionary...")
    if data_name == 'blogs':
        try:
            with open('./blogs'+str(MAX_LENGTH)+'.json', 'r') as jsonfile:
                dictionary = json.load(jsonfile)
        except:
            with open(data_dir, 'r') as f:
                sentence_list = f.readlines()
            dictionary = blogsToDictionary(sentence_list, MAX_LENGTH)
    elif data_name == 'twitter':
        try:
            with open('./twitter'+str(MAX_LENGTH)+'.json', 'r') as jsonfile:
                dictionary = json.load(jsonfile)
        except:
            with open(data_dir, 'r') as f:
                sentence_list = f.readlines()
            dictionary = twitterToDictionary(sentence_list, MAX_LENGTH)
    #dictionary: {'tag': {'tokens': [], 'sentences': []}}
    print("Chaning to Variables...")
    train_data, test_data = splitTrainAndTestData(dictionary)
    train_pair, test_pair, test_sentence = getPairs(train_data, test_data)
    train_input_var, train_output_label, test_input_var, test_output_label,\
    train_input = changeToVariables(train_pair, test_pair, CUDA_use, data_name,
    MAX_VOCAB)
    print("Data Preparation Done.")
    return train_input_var, train_output_label, test_input_var,\
    test_output_label, test_sentence, train_input

"""
if __name__ == "__main__":
    twitter_data_dir = '/Users/jaeickbae/Documents/projects/'+\
    '2017 Affective Computing/Jan9-2012-tweets-clean.txt'
    blogs_data_dir = '/Users/jaeickbae/Documents/projects/2017 Affective Computing/Emotion-Data/Benchmark/category_gold_std.txt'
    with open(blogs_data_dir, 'r') as f:
        lines = f.readlines()
    prepareData(lines, 'blogs', blogs_data_dir, MAX_LENGTH=30)
"""
