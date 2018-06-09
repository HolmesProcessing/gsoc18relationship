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

        self.feature_length = feature_length
        self.label_length = label_length
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

        print('Initial shape: {0}'.format(self.X_train.shape))
        print('Resulting shape: {0}'.format(X_res.shape))
        print('Initial dataset shape {}'.format(Counter(self.y_train)))
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        self.X_train = X_res
        self.y_train = y_res

    def prepare_train_bin(self):
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
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=labels))

        return (y_out, loss)

    def build(self, learning_rate):
        self.train_features = tf.placeholder(tf.float32, shape=(None, self.feature_length))
        self.train_labels = tf.placeholder(tf.int8, shape=(None, self.label_length))

        self.y_out, self.loss = self.build_NN(self.train_features, self.train_labels)
        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.new_features = tf.placeholder(tf.float32, shape=(None, self.feature_length))
        self.new_labels = tf.placeholder(tf.int8, shape=(None, self.label_length))
        self.new_y_out, self.new_loss = self.build_NN(features=self.new_features, labels=self.new_labels)

        self.init_op = tf.global_variables_initializer()

    def train(self, iteration, num_epochs=200):
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        for epoch in range(num_epochs):
            print('Epoch %2d/%2d: ' % (epoch + 1, num_epochs))

            self.sess.run(self.train_opt, feed_dict={self.train_features:self.X_train, self.train_labels:self.y_train_bin})

            y_out = self.predict(self.X_train)
            train_loss = self.evaluate(self.X_train, self.y_train_bin)
            train_acc = self.get_accuracy(y_out, self.y_train_bin)
            msg = ' loss = %8.4f, acc = %3.2f%%' % (train_loss, train_acc * 100)
            print(msg)

    def predict(self, X):
        return self.sess.run(self.new_y_out, feed_dict={self.new_features: X})

    def evaluate(self, X, y):
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X, self.new_labels: y})

    def get_accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


if __name__ == '__main__':
    nn_instance = NN(FILE_LOCATION, feature_length=FEATURE_LENGTH, label_length=LABEL_LENGTH)

    i = 0

    nn_instance.split_train_test(0.4, 0)
    nn_instance.resample(42)
    nn_instance.prepare_train_bin()

    nn_instance.train(iteration=i)
