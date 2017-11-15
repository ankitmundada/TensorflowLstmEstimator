# Author: Ankit Mundada
# Date: 7/11/2017
import os.path
import numpy as np
import collections
import json


FLAGS = {
    'WORD_TO_INDEX': './word_to_index.json',
    'INDEX_TO_WORD': './index_to_word.json',
    'TRAINING_DATA': './datasets/train.txt',
    'VAL_DATA': './datasets/val.txt',
    'TEST_DATA': './datasets/test.txt',
}
word_idx, idx_word = None, None


def _read_input_into_list(filename):
    with open(filename) as f:
        list_words = f.readlines()
    list_words = [x.strip() for x in list_words]
    list_words = [list_words[i].split() for i in range(len(list_words))]
    list_words = np.array(list_words)
    list_words = np.reshape(list_words, [-1, ])
    return list_words


def _build_vocab(list_words):
    count = collections.Counter(list_words).most_common()
    vocab = dict()
    for word, _ in count:
        vocab[word] = len(vocab)
    reverse = dict(zip(vocab.values(), vocab.keys()))
    return vocab, reverse


def convert_word_to_index(data):
    return [word_idx[word] for word in data]


def convert_index_to_word(data):
    return [idx_word[str(idx)] for idx in data]


def initiate_vocabs(is_forced=False):
    """
    Analyze the text corpus and save all the unique words into a json vocabulary. (This happens only first time). If the
    corpus is modified, delete the original vocabulary, so that this method will re-create it with modified corpus.)
    :return: Size of the vocabulary
    """
    global word_idx
    global idx_word
    if not os.path.exists(FLAGS['INDEX_TO_WORD']) or not os.path.exists(FLAGS['WORD_TO_INDEX']) or is_forced:
        input_word_list = _read_input_into_list(FLAGS['TRAINING_DATA'])
        np.append(input_word_list, _read_input_into_list(FLAGS['VAL_DATA']))
        np.append(input_word_list, _read_input_into_list(FLAGS['TEST_DATA']))
        word_idx, idx_word = _build_vocab(input_word_list)
        with open(FLAGS['WORD_TO_INDEX'], 'w') as outfile:
            json.dump(word_idx, outfile)
        with open(FLAGS['INDEX_TO_WORD'], 'w') as outfile:
            json.dump(idx_word, outfile)
    else:
        with open(FLAGS['WORD_TO_INDEX'], 'r') as infile:
            word_idx = json.load(infile)
        with open(FLAGS['INDEX_TO_WORD'], 'r') as infile:
            idx_word = json.load(infile)
    size_vocab = len(idx_word)
    return size_vocab


# convert a text file into a csv file with each row as input for the model
def make_csv(filepath, context_size):
    expected_file_name = filepath + '_context_' + str(context_size)
    if not os.path.exists(expected_file_name):
        with open(expected_file_name, 'w') as file:
            word_list = _read_input_into_list(filepath)
            word_list = convert_word_to_index(word_list)
            offset = 0
            while offset + context_size + 1 < len(word_list):
                input_slice = word_list[offset:offset + context_size + 1]
                row = ''
                for idx in input_slice:
                    row += str(idx) + ','
                row = row[:-1] + '\n'
                file.write(row)
                offset += 1

    return expected_file_name
