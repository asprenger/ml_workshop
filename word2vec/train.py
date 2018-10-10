# -*- coding: utf-8 -*-
"""
Train word2vec model.
"""
import os
import datetime
import random
import time
import h5py
import argparse
import numpy as np
from gensim.corpora import Dictionary

from keras.layers import Input, Embedding, Lambda, Dense
from keras.layers import Concatenate, Average, Add
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop

def current_time_ms():
    return int(time.time()*1000.0)

def ts_rand():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_num = random.randint(1e6, 1e7-1)
    return '%s_%d' % (ts, random_num)


def run(dataset_path, vocab_path, model_dir):

    print('Load vocabulary')
    vocab = Dictionary.load(vocab_path)
    vocab_size = len(vocab.token2id)
    print('vocab_size:', vocab_size)

    print('Load', dataset_path)
    f = h5py.File(dataset_path, 'r')

    X_train = f['x_train'].value
    y_train = f['y_train'].value

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)

    max_train_size = 10000000
    print('Cutoff train data to %d examples' % max_train_size)
    X_train = X_train[:max_train_size,:]
    y_train = y_train[:max_train_size]

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)

    print('Shuffle dataset')
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    vec_dims = 100
    aggregation_type = 'concat' # 'average', `sum` or 'concat'
    win_size = X_train.shape[1]
    print('win_size:', win_size)

    # Define input shape
    inputs = Input(shape=(win_size,), dtype='int32')

    # Embedding layer maps each tokenId in the window to a `vec_dim` dim.
    # vector. Output shape: (-1, win_size, vec_dim)
    word_vectors = Embedding(vocab_size, vec_dims)(inputs)

    # The embedding layer output is fed into `win_size` lambda layers that 
    # each outputs a word vector.
    sliced_word_vector = [Lambda(lambda x: x[:,i,:], output_shape=(vec_dims,))(word_vectors) for i in range(win_size)]

    # Aggregate the word vectors
    if aggregation_type == 'concat':
        h = Concatenate()(sliced_word_vector)
    elif aggregation_type == 'average':
        h = Average()(sliced_word_vector)
    elif aggregation_type == 'sum':
        h = Add()(sliced_word_vector)
    else:
        raise ValueError('Invalid row aggregation')
        
    # Feed the aggregated word vectors into a dense layer that returns
    # a probability for each word in the vocabulary    
    probs = Dense(vocab_size, activation='softmax')(h)  

    model = Model(inputs, probs)

    # Compile the model with SGD optimizer and cross entropy loss
    model.compile(optimizer=SGD(lr=0.05),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    epochs_per_fit = 1

    for i in range(100):

        start = current_time_ms()
        h = model.fit(X_train, y_train,
              batch_size=256,
              epochs=epochs_per_fit,
              verbose=0)
        mean_epoch_time = (current_time_ms() - start) / epochs_per_fit

        epoch = i * epochs_per_fit
        loss = h.history['loss'][-1]
        acc = h.history['acc'][-1]

        print('epoch %d: loss=%f acc=%f time=%f' % (epoch, loss, acc, mean_epoch_time))

        checkpoint_dir = os.path.join(model_dir, ts_rand())
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'w2v_model.h5')
        print('Save model:', checkpoint_path)
        model.save(checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='Train dataset path', default='./dataset.hdf5')
    parser.add_argument('--vocab-path', help='Vocabulary path', default='./vocab.pkl')
    parser.add_argument('--models-path', help='Output directory', default='/tmp/w2v_models')
    args = parser.parse_args()
    run(dataset_path=args.dataset_path, vocab_path=args.vocab_path, model_dir=args.models_path)
