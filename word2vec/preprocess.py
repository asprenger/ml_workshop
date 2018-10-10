# -*- coding: utf-8 -*-
"""
Prepare a set of articles for training a word2vec model.

The following output files are generated:

    ${output_path}/vocab.pkl - A Gensim dictionary in binary format
    ${output_path}/vocab.txt - A Gensim dictionary in text format
    ${output_path}/dataset.hdf5 - A hdf5 file with keys `x_train` and `y_train`
"""
import os
import shutil
import re
import codecs
import argparse
from itertools import islice, chain
import h5py
import numpy as np
from gensim.corpora import Dictionary

def delete_dir(path):
    shutil.rmtree(path, ignore_errors=True)

def read_lines(path):
    '''Return lines in file'''
    return [line.strip() for line in codecs.open(path, "r", "utf-8")]

def load_stopwords(stopwords_path):
    print("Loading stopwords: %s", stopwords_path)
    stopwords = read_lines(stopwords_path)
    return dict(map(lambda w: (w.lower(), ''), stopwords))

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def normalize_text(text):
    '''Convert text to lower-case, remove HTML tags and pad punctuation with spaces.'''
    norm_text = text.lower()
    # replace HTML tags
    norm_text = re.sub(r'<[^>]+>', ' ', norm_text)
    # replace links
    norm_text = re.sub('http[s]?://\S*', ' ', norm_text)
    # replace integer and float numbers
    norm_text = re.sub('[+-]?([0-9]*[,])?[0-9]+', ' ', norm_text)
    # replace HTML encoded characters
    html_encodings = ['&quot;', '&amp;', '&nbsp;', '&lt;', '&gt;']
    html_replacements = ['"', '&', ' ', '<', '>']
    for a,b in zip(html_encodings, html_replacements):
        norm_text = norm_text.replace(a, b)
    # remove punctuations
    for char in ['.', '"', '+', '-', '–', '*',  '\'', '/', ',', '(', ')', '[', ']', '{', '}', '!', '?', ';', ':', '«', '»', '„', '“', '…', '»', '«', '#', '%', '$']:
        norm_text = norm_text.replace(char, ' ')
    return norm_text

def run(input_path, output_path, stopword_path, win_size, vocab_size):

    os.makedirs(output_path, exist_ok=True)
    
    newline = '\n'.encode("utf-8")
    stopwords = load_stopwords(stopword_path)

    print('Build vocabulary')
    vocab = Dictionary(prune_at=None)
    with open(input_path, encoding="utf-8") as input_file:
        for i, line in enumerate(input_file):
            if i % 1000 == 0 and i != 0:
                print('%d articles added to dictionary' % i)
            line = line.strip()
            txt_norm = normalize_text(line)
            tokens = txt_norm.split()
            tokens = [token for token in tokens if len(token) > 1 and token not in stopwords]
            vocab.add_documents([tokens], prune_at=None)

    print('num words:', len(vocab.token2id))
    print('num_documents:', vocab.num_docs)
    vocab.filter_extremes(no_below=20, no_above=1.0, keep_n=vocab_size)
    print('num words:', len(vocab.token2id))
    print('num_documents:', vocab.num_docs)

    vocab.save_as_text(os.path.join(output_path, 'vocab.txt'))
    vocab.save(os.path.join(output_path, 'vocab.pkl'))
    
    # min. number of tokens in a doc for the final dataset
    min_doc_size = 100

    example_size = win_size - 1

    # Tokenize the text and build examples
    dataset_path = os.path.join(output_path, 'dataset.hdf5')
    with h5py.File(dataset_path, 'w') as data_file:

        x_train_ds = None
        y_train_ds = None

        with open(input_path, encoding="utf-8") as input_file:
            for i, line in enumerate(input_file):

                if i % 1000 == 0 and i != 0:
                    print('%d articles tokenized' % i)

                # cleanup the text
                line = line.strip()
                txt_norm = normalize_text(line)

                # tokenize text
                tokens = txt_norm.split()

                # filter tokens and convert to tokenIDs
                tokens = [vocab.token2id[token] for token in tokens if len(token) > 1 and token not in stopwords and token in vocab.token2id]
                
                # skip documents that are to short to generate a context or that are 
                # shorter than `min_doc_size`
                if len(tokens) < win_size + 1 or len(tokens) < min_doc_size:
                    continue

                # create examples and labels by sliding window over text
                examples = []
                labels = []
                for win in window(tokens, n=win_size):
                    mid_pos = int(win_size / 2)
                    example = list(chain(win[:mid_pos], win[mid_pos+1:]))
                    examples.append(example)
                    label = win[mid_pos]
                    labels.append(label)

                # write to file
                x_train = np.array(examples, dtype=np.int32)
                y_train = np.array(labels, dtype=np.int32)

                if x_train_ds == None:
                    x_train_ds = data_file.create_dataset('x_train', 
                                                          data=x_train, 
                                                          chunks=(10000, example_size), 
                                                          maxshape=(None, example_size))
                    y_train_ds = data_file.create_dataset('y_train', 
                                                  data=y_train, 
                                                  chunks=(10000,), 
                                                  maxshape=(None,))
                else:
                    nb_examples = x_train.shape[0]    
                    x_train_ds.resize(x_train_ds.shape[0] + nb_examples, axis=0)
                    x_train_ds[-nb_examples:,:] = x_train
                    y_train_ds.resize(y_train_ds.shape[0] + nb_examples, axis=0)
                    y_train_ds[-nb_examples:] = y_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='Input text file, each line is an article', default='./enwiki.txt')
    parser.add_argument('--stopword-path', help='Stopwords path', default='./stopwords_english.txt')
    parser.add_argument('--output-path', help='Output directory', default='./')
    parser.add_argument('--win-size', help='Window size', type=int, default=11)
    parser.add_argument('--vocab-size', help='Size of vocabulary', type=int, default=10000)
    args = parser.parse_args()
    run(input_path=args.input_path, output_path=args.output_path, win_size=args.win_size, 
        stopword_path=args.stopword_path, vocab_size=args.vocab_size)

if __name__ == '__main__':
    main()
