# -*- coding: utf-8 -*-
"""
"""
import os
import datetime
import random
import time
import h5py
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


def main():

    checkpoint_base_dir = '/tmp/w2v_models'

    print('Load vocabulary')
    vocab = Dictionary.load('dewiki_vocab=10000/vocab.pkl')
    vocab_size = len(vocab.token2id)
    print('vocab_size:', vocab_size)

    filename = 'dewiki_vocab=10000/dataset.hdf5'
    print('Load ', filename)
    f = h5py.File(filename, 'r')

    X_train = f['x_train'].value
    y_train = f['y_train'].value

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)

    print('Cutoff train data')
    X_train = X_train[:10000000,:]
    y_train = y_train[:10000000]

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

    # TODO refactor this to a for loop, that is more readable
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

        checkpoint_dir = os.path.join(checkpoint_base_dir, ts_rand())
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'w2v_model.h5')
        print('Save model:', checkpoint_path)
        model.save(checkpoint_path)


if __name__ == '__main__':
    main()
