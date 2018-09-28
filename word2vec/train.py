import numpy as np
from gensim.corpora import Dictionary
from keras.layers import Input, Embedding, Lambda, Dense
from keras.layers import Concatenate, Average, Add
from keras.models import Model
from keras.optimizers import SGD

def main():

    vocab = Dictionary.load('vocab.pkl')
    vocab_size = len(vocab.token2id)
    X = np.load('examples.npy')
    y = np.load('labels.npy')

    vec_dims = 10
    aggregation_type = 'concat' # 'average', `sum` or 'concat'
    win_size = X.shape[1]

    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    print('vocab_size:', vocab_size)
    print('win_size:', win_size)

    # Define input shape. Each example is a `win_size` dim. vector
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
    model.compile(optimizer=SGD(lr=0.01),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X, y,
          batch_size=4,
          epochs=100,
          verbose=1)

    samples = np.array([
        [ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10],
        [ 7,  8,  9, 10, 11, 13, 14, 15, 16, 17],
        [15, 16, 17, 18, 19, 21, 22, 23, 24, 25]
        ])
    probs = model.predict(samples)
    print(np.argmax(probs, axis=1))

if __name__ == '__main__':
    main()
