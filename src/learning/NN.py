import pickle
import random
import numpy as np
import tensorflow as tf
import feed_handling_pb2

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
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
    def __init__(self, file_path, feature_length, label_length, learning_rate=1e-3):
        self.X_cuckoo = []
        self.X_objdump = []
        self.X_peinfo = []
        self.X_richheader = []
        self.y = []

        objects = pickle.load(open(file_path, 'rb'))

        reader = feed_handling_pb2.TrainingData()
        for o in objects:
            reader.ParseFromString(o)

            if not reader.features_cuckoo:
                self.X_cuckoo.append(['nop'] * 100)
            else:
                self.X_cuckoo.append(reader.features_cuckoo)

            if not reader.features_objdump:
                self.X_objdump.append(np.zeros(100))
            else:
                self.X_objdump.append(np.array(map(float, reader.features_objdump)))

            if not reader.features_peinfo:
                self.X_peinfo.append(np.zeros(17))
            else:
                self.X_peinfo.append(np.array(map(float, reader.features_peinfo)))

            if not reader.features_richheader:
                self.X_richheader.append(np.array([[0,0,0]] * 20))
            else:
                self.X_richheader.append(np.array(reader.features_richheader))

            self.y.append(encode_label(reader.label))

        self.X_objdump = preprocessing.MinMaxScaler().fit_transform(self.X_objdump)
        self.X_peinfo = preprocessing.MinMaxScaler().fit_transform(self.X_peinfo)
        self.X = np.concatenate((self.X_objdump, self.X_peinfo), axis=1)
        self.y = np.array(self.y)

        self.feature_length = feature_length
        self.label_length = label_length
        self.W_out = None
        self.b_out = None

        tf.set_random_seed(1337)
        self.build(learning_rate)

    def split_train_test(self, num_splits, random_state):
        skf = StratifiedKFold(n_splits=num_splits, random_state=random_state)
        return skf.split(self.X, self.y)

    def resample(self, random_state):
        ros = RandomUnderSampler(random_state=random_state)
        X_res, y_res = ros.fit_sample(self.X_train, self.y_train)
        X_res, y_res = shuffle(X_res, y_res, random_state=0)

        print('Initial shape: {0}'.format(self.X_train.shape))
        print('Resulting shape: {0}'.format(X_res.shape))
        print('Initial dataset shape {}'.format(Counter(self.y_train)))
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        self.X_train = X_res
        self.y_train = y_res

    def prepare_data(self, train_index, test_index):
        self.X_train, self.X_test = self.X[train_index], self.X[test_index]
        self.y_train, self.y_test = self.y[train_index], self.y[test_index]

        num_y = np.size(self.y_train)
        self.y_train_bin = np.zeros((num_y, self.label_length))

        for i in range(num_y):
            self.y_train_bin[i, self.y_train[i]] = 1

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

        y_raw = tf.matmul(h_2, self.W_out) + self.b_out
        y_out = tf.nn.softmax(y_raw)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=labels))

        return (y_out, loss)

    def build(self, learning_rate):
        self.train_features = tf.placeholder(tf.float32, shape=(None, self.feature_length))
        self.train_labels = tf.placeholder(tf.int8, shape=(None, self.label_length))
        self.y_out, self.loss = self.build_NN(self.train_features, self.train_labels)

        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.train_labels, 1))
        self.correct_predictions = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.init_op = tf.global_variables_initializer()

    def train(self, num_epochs=200):
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        for epoch in range(num_epochs):
            print('Epoch %2d/%2d: ' % (epoch + 1, num_epochs))

            self.sess.run(self.train_opt, feed_dict={self.train_features:self.X_train, self.train_labels:self.y_train_bin})

            train_loss = self.evaluate(self.X_train, self.y_train_bin)
            train_acc = self.get_accuracy(self.X_train, self.y_train_bin)
            print(' loss = %8.4f, acc = %3.2f%%' % (train_loss, train_acc * 100))

    def evaluate(self, X, y):
        return self.loss.eval(feed_dict={self.train_features:X, self.train_labels:y}, session=self.sess)

    def get_accuracy(self, X, y):
        return self.correct_predictions.eval(feed_dict={self.train_features:self.X_train, self.train_labels:self.y_train_bin}, session=self.sess) / X.shape[0]


if __name__ == '__main__':
    nn_instance = NN(FILE_LOCATION, feature_length=FEATURE_LENGTH, label_length=LABEL_LENGTH)

    skf = nn_instance.split_train_test(3, 0)

    for train_index, test_index in skf:
        nn_instance.prepare_data(train_index, test_index)
        nn_instance.train()
