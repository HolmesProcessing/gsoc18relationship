import pickle
import random
import numpy as np
import tensorflow as tf
import feed_handling_pb2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def encode_label(label):
    labels = ['CryptoRansom', 'apt1', 'athena_variant', 'betabot', 'blackshades', 'citadel_krebs', 'darkcomet', 'darkddoser', 'dirtjumper', 'expiro', 'gamarue', 'ghostheart2', 'locker', 'machbot', 'mediyes', 'nitol', 'pushdo', 'shylock', 'simda', 'yoyoddos2']
    return labels.index(label)

class NN:
    def __init__(self, file_path, feature_length, label_length, model='FFNN', learning_rate=1e-3):
        reader = feed_handling_pb2.TrainingData()
        data = pickle.load(open(file_path, 'rb'))

        self.X = []
        self.y = []

        for d in data:
            reader.ParseFromString(d)
            self.X.append(np.array(map(float, reader.features)))
            self.y.append(reader.label)

        self.X = preprocessing.MinMaxScaler().fit_transform(self.X)
        self.y = np.array(self.y)

        self.feature_length = feature_length
        self.label_length = label_length
        self.model = model

        self.W_out = None
        self.b_out = None

        tf.set_random_seed(1337)
        self.build(learning_rate)

    def split_train_test(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)

    def resample(self, random_state):
        ros = RandomUnderSampler(random_state=random_state)
        X_res, y_res = ros.fit_sample(self.X_train, self.y_train)
        X_res, y_res = shuffle(X_res, y_res, random_state=0)

        print "Initial shape: {0}".format(self.X_train.shape)
        print "Resulting shape: {0}".format(X_res.shape)
        print('Initial dataset shape {}'.format(Counter(self.y_train)))
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        self.X_train = X_res
        self.y_train = y_res

    def prepare_train_bin(self):
        num_y = np.size(self.y_train)
        self.y_train_bin = np.zeros((num_y, self.label_length))

        for i in range(num_y):
            self.y_train_bin[i, encode_label(self.y_train[i])] = 1

    def load_dataset_train(self, batch_size):
        for i in range(0, len(self.X_train) - batch_size + 1, batch_size):
            excerpt = slice(i, i + batch_size)

            yield self.X_train[excerpt], self.y_train_bin[excerpt]

    def build_NN(self, features, labels):
        if not self.W_out:
            self.W_out = weight_variable([self.feature_length, self.label_length])

        if not self.b_out:
            self.b_out = bias_variable([self.label_length])

        W_ff_1 = weight_variable([self.feature_length, self.feature_length])
        b_ff_1 = bias_variable([self.feature_length])
        W_ff_2 = weight_variable([self.feature_length, self.feature_length])
        b_ff_2 = bias_variable([self.feature_length])

        h = tf.nn.relu(tf.matmul(features, W_ff_1) + b_ff_1)
        h_2 = tf.nn.relu(tf.matmul(h, W_ff_2) + b_ff_2)
        h_out = tf.nn.dropout(h_2, self.keep_prob)

        y_raw = tf.matmul(h_2, self.W_out) + self.b_out
        y_out = tf.nn.softmax(y_raw)

        self.grad_ffnn = tf.gradients(y_out, features)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=labels))
        self.labels = tf.argmax(y_out, 1)

    def build(self, learning_rate):
        self.train_features = tf.placeholder("float32", shape=(None, self.feature_length))
        self.train_labels = tf.placeholder("float32", shape=(None, self.label_length))
        self.keep_prob = tf.placeholder("float")

        self.build_NN(self.train_features, self.train_labels)
        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.init_op = tf.global_variables_initializer()

    def train(self, iteration, num_epochs=200):
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0

            for batch_id, batch in enumerate(self.load_dataset_train(20)):
                X, y = batch

                self.sess.run(self.train_opt, feed_dict={self.train_features:X, self.train_labels:np.float64(y), self.keep_prob: 0.7})
                train_err_new = self.loss.eval(feed_dict={self.train_features:X, self.train_labels:y, self.keep_prob: 1.0}, session=self.sess)

                print("{0}: {1}".format(batch_id, train_err_new))
                train_err += train_err_new

                print("Epoch: {0}, Cost: {1}".format(epoch, train_err))

    def predict(self, X):
        pass

if __name__ == '__main__':
    nn_instance = NN(FILE_LOCATION, feature_length=FEATURE_LENGTH, label_length=LABEL_LENGTH)

    i = 0

    nn_instance.split_train_test(0.4, 0)
    nn_instance.resample(42)
    nn_instance.prepare_train_bin()

    nn_instance.train(iteration=i)
