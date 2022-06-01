import queue
import tensorflow as tf
import numpy as np # version 1.16
import re
import os
import json
from scipy.sparse import csc_array, csc_matrix
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

CONFIG = "./.conf.json"


tf.compat.v1.disable_eager_execution()
filts = re.compile(r'[a-z]+')
# corpus = 'My name is yxy. yxy is me!'.lower()
corpus = ' '.join(open("./THE TRAGEDY OF ROMEO AND JULIET.txt", mode='r').readlines()).lower()

words = filts.findall(corpus)
words = set(words) 
word2int = {}
int2word = {}
vocab_size = len(words)

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw sentences is a list of sentences.
raw_sentences = corpus.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(filts.findall(sentence))
WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

indices_y = [i for i in range(vocab_size)]
indices_x_of_x_train = []
indices_x_of_y_train = []
cnt = 0
for data_word in data:
    indices_x_of_x_train.append([cnt, word2int[ data_word[0] ]])
    indices_x_of_y_train.append([cnt, word2int[ data_word[1]]])
    cnt += 1
# print(len(indices_x_of_x_train),len(indices_x_of_y_train), vocab_size)
vals_x = [1 for i in range(len(indices_x_of_x_train))]
vals_y = [1 for i in range(len(indices_x_of_y_train))]
x_train = tf.compat.v1.SparseTensorValue(indices=indices_x_of_x_train, values=[1 for _ in range(len(indices_x_of_x_train))], dense_shape=[len(indices_x_of_x_train), vocab_size])
y_train = tf.compat.v1.SparseTensorValue(indices=indices_x_of_y_train, values=[1 for _ in range(len(indices_x_of_x_train))], dense_shape=[len(indices_x_of_y_train), vocab_size])
# print(x_train, y_train)

# print("x")
# input()
# making placeholders for x_train and y_train
# x = tf.compat.v1.placeholder(tf.float32, shape=(None, vocab_size))
# y_label = tf.compat.v1.placeholder(tf.float32, shape=(None, vocab_size))
x = tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, vocab_size))
EMBEDDING_DIM = 5 # you can choose your own number
W1 = tf.Variable(tf.compat.v1.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.compat.v1.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.sparse.sparse_dense_matmul(x,W1), b1)

W2 = tf.Variable(tf.compat.v1.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.compat.v1.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init) #make sure you do this!

# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.sparse.reduce_sum(y_label * tf.math.log(prediction), axis=[1]))

# define the training step:
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000
# train for n_iter iteration
print('fine')

for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    # print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))
def dist_between():
    dists = {}
    for index in range(len(int2word)):
        for another_index in range(index+1, len(int2word)):
            dists[str(index) + '-' + str(another_index)] = float(euclidean_dist(vectors[index], vectors[another_index]))
    return dists
json.dump({
'word2int': word2int,
'vectors' : vectors
}, open("./.conf.json", mode='w'), cls=NumpyEncoder)
json.dump({
    'dist': dist_between()
})