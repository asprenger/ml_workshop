from numpy import dot
from gensim import matutils
from gensim.corpora import Dictionary
from tensorflow.keras.models import Model, load_model

def cosine_similarity(doc_vec1, doc_vec2):
    # Taken from: gensim.models.keyedvectors.Doc2VecKeyedVectors    
    return dot(matutils.unitvec(doc_vec1), matutils.unitvec(doc_vec2))

def token_vec(token, vocab, W):
    token_id = vocab.token2id[token]
    return W[token_id]

def most_similar(token, vocab, W, topn=5):
    vec = token_vec(token, vocab, W)
    dists = [cosine_similarity(vec, W[i]) for i in range(W.shape[0])]
    return matutils.argsort(dists, topn, reverse=True)


vocab = Dictionary.load('/home/ubuntu/enwiki_vocab=10000/vocab.pkl')

model_path = '/tmp/w2v_models/20180928_061054_4305059/w2v_model.h5'
model = load_model(model_path)

embedding_layer = model.layers[1]
print(embedding_layer)

W = embedding_layer.get_weights()[0]
print(W.shape)

tokens = ['london', 'berlin', 'paris', 'blue', 'red', 'yellow']

vecs = [token_vec(token, vocab, W) for token in tokens]

print(cosine_similarity(vecs[0],vecs[1]))
print(cosine_similarity(vecs[0],vecs[2]))
print(cosine_similarity(vecs[1],vecs[2]))

print(cosine_similarity(vecs[3],vecs[4]))
print(cosine_similarity(vecs[3],vecs[5]))
print(cosine_similarity(vecs[4],vecs[5]))

print(vocab[468])

foo = most_similar('london', vocab, W, topn=5)
for i in foo:
    print(vocab[i], i)

foo = most_similar('blue', vocab, W, topn=5)
for i in foo:
    print(vocab[i], i)
