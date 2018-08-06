import sys
sys.path.append('../')

import pickle
import random
import time
import glob
import os
import numpy as np
import tensorflow as tf
import feedhandling.feed_handling_pb2 as feed_handling_pb2

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

def max_pool_2(x):
    return tf.nn.avg_pool(x, ksize=[1, 76, 1, 1], strides=[1, 76, 1, 1], padding='SAME')

def encode_label(labels):
    label = ['CryptoRansom', 'apt1', 'athena_variant', 'betabot', 'blackshades', 'citadel_krebs', 'darkcomet', 'darkddoser', 'dirtjumper', 'expiro', 'gamarue', 'ghostheart2', 'locker', 'machbot', 'mediyes', 'nitol', 'pushdo', 'shylock', 'simda', 'yoyoddos2']

    switcher = {
        'athena_variant': 20,   # CIA Malware
        'betabot': 21,          # Hijacker
        'blackshades': 22,      # Trojan
        'darkcomet': 22,
        'nitol': 22,
        'shylock': 22,
        'citadel_krebs':23,     # Zeus offshoot
        'darkdoser':24,         # Password stealing tool
        'dirtjumper':25,        # DDoS Bot
        'expiro': 26,           # Virus
        'gamarue': 27,          # Worm
        'machbot': 28,          # Botnet
        'pushdo': 28,
        'simda': 28,
        'yoyoddos2': 28
    }

    classifications = []

    for l in labels:
        classifications.append(label.index(l))

    if len(classifications) == 1:
        classifications.append(switcher.get(label[classifications[0]], 29))

    return classifications

def get_latest_model():
    models = glob.glob('./models/*')
    return max(models, key=os.path.getctime)

class NN:
    def __init__(self, file_path, labels_length, learning_rate=0.01):
        X_cuckoo = []
        X_objdump = []
        X_peinfo = []
        X_richheader = []

        y = []
        objects = pickle.load(open(file_path, 'rb'))

        reader = feed_handling_pb2.TrainingData()
        for o in objects:
            reader.ParseFromString(o)

            try:
                y.append(encode_label(reader.labels))
            except:
                continue

            if not reader.features_cuckoo:
                X_cuckoo.append(np.zeros(150))
            else:
                X_cuckoo.append(np.array(map(int, reader.features_cuckoo)))

            if not reader.features_objdump:
                X_objdump.append(np.zeros(100))
            else:
                X_objdump.append(np.array(map(float, reader.features_objdump)))

            if not reader.features_peinfo:
                X_peinfo.append(np.zeros(17))
            else:
                X_peinfo.append(np.array(map(float, reader.features_peinfo)))

            if not reader.features_richheader:
                X_richheader.append(np.zeros(80))
            else:
                X_richheader.append(np.array(map(float, reader.features_richheader)))

        X_cuckoo = np.array(X_cuckoo).astype(np.int32)
        X_objdump = preprocessing.MinMaxScaler().fit_transform(X_objdump)
        X_peinfo = preprocessing.MinMaxScaler().fit_transform(X_peinfo)
        X_richheader = preprocessing.MinMaxScaler().fit_transform(X_richheader)

        mlp_features = np.concatenate((X_objdump, X_peinfo, X_richheader), axis=1).astype(np.float32)

        self.X = np.concatenate((X_cuckoo, mlp_features), axis=1)
        y = np.array(y)
        self.y_enc = np.add(np.multiply(y[:,0], 100), y[:,1])

        self.labels_length = labels_length
        self.learning_rate = learning_rate
        self.W_out = None
        self.b_out = None

        tf.set_random_seed(1337)

    def split_train_test(self, num_splits, random_state):
        skf = StratifiedKFold(n_splits=num_splits, random_state=random_state)
        return skf.split(self.X, self.y_enc)

    def resample_training_data(self, random_state):
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_sample(self.X_train, self.y_train_enc)
        X_res, y_res = shuffle(X_res, y_res, random_state=0)

        print('Initial shape: {0}'.format(self.X_train.shape))
        print('Resulting shape: {0}'.format(X_res.shape))
        print('Initial dataset shape {}'.format(Counter(self.y_train_enc)))
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        self.X_train = X_res
        self.y_train_enc = y_res

    def prepare_data(self, train_index, test_index):
        self.X_train, self.X_test = self.X[train_index], self.X[test_index]
        self.y_train_enc, self.y_test_enc = self.y_enc[train_index], self.y_enc[test_index]

        self.resample_training_data(42)

        def decode(c, d):
            return ((c - c % d) / d), (c % d)

        vf = np.vectorize(decode)
        self.y_train = np.concatenate(vf(self.y_train_enc.reshape(-1, 1), 100), axis=1)
        self.y_test = np.concatenate(vf(self.y_test_enc.reshape(-1, 1), 100), axis=1)

        num_y_train = self.y_train.shape[0]
        self.y_train_bin = np.zeros((num_y_train, self.labels_length))

        for i in range(num_y_train):
            self.y_train_bin[i, self.y_train[i][0]] = 1

            if self.y_train[i][1] != 29:
                self.y_train_bin[i, self.y_train[i][1]] = 1

        num_y_test = self.y_test.shape[0]
        self.y_test_bin = np.zeros((num_y_test, self.labels_length))

        for i in range(num_y_test):
            self.y_test_bin[i, self.y_test[i][0]] = 1

            if self.y_test[i][1] != 29:
                self.y_test_bin[i, self.y_test[i][1]] = 1

    def build_mlp(self, features, feature_length):
        W_ff_1 = weight_variable([feature_length, feature_length])
        b_ff_1 = bias_variable([feature_length])
        W_ff_2 = weight_variable([feature_length, feature_length])
        b_ff_2 = bias_variable([feature_length])

        h = tf.nn.relu(tf.matmul(features, W_ff_1) + b_ff_1)
        h_2 = tf.nn.relu(tf.matmul(h, W_ff_2) + b_ff_2)

        return h_2

    def build_cnn(self, features, dict_length, embedded_length):
        W = tf.Variable(tf.random_uniform([dict_length, embedded_length], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, features)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        W_conv_1 = weight_variable([3, embedded_length, 1, 3])
        b_conv_1 = bias_variable([3])
        W_conv_2 = weight_variable([3, embedded_length, 3, 6])
        b_conv_2 = bias_variable([6])

        h_conv_1 = tf.nn.relu(conv2d(embedded_chars_expanded, W_conv_1) + b_conv_1)
        h_pool_1 = max_pool_1(h_conv_1)
        h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
        h_pool_2 = max_pool_2(h_conv_2)

        return tf.reshape(h_pool_2, [-1, embedded_length * 6])

    def build(self):
        self.X_mlp_features = tf.placeholder(tf.float32, shape=(None, 197), name='X_mlp_features')
        self.X_cnn_features = tf.placeholder(tf.int32, shape=(None, 150), name='X_cnn_features')
        self.y_labels = tf.placeholder(tf.float32, shape=(None, self.labels_length), name='y_labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        NN_mlp = self.build_mlp(self.X_mlp_features, 197)
        NN_cnn = self.build_cnn(self.X_cnn_features, 322, 10)
        h_concat = tf.concat([NN_mlp, NN_cnn], 1)
        h_dropout = tf.nn.dropout(h_concat, self.keep_prob)

        if not self.W_out:
            self.W_out = weight_variable([257, self.labels_length])

        if not self.b_out:
            self.b_out = bias_variable([self.labels_length])

        self.y_raw = tf.nn.bias_add(tf.matmul(h_dropout, self.W_out), self.b_out, name='y_raw')
        y_out = tf.nn.sigmoid(self.y_raw)
        self.labels = tf.round(y_out, name='labels')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_raw, labels=self.y_labels), name='loss')

        self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(self.labels, tf.round(self.y_labels))
        correct_predictions = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
        self.accuracy = tf.reduce_mean(correct_predictions, name='accuracy')

        self.init_op = tf.global_variables_initializer()

    def load_dataset_train(self, batchsize):
        for start_idx in range(0, self.X_train.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield self.X_train[excerpt,:], self.y_train_bin[excerpt,:]

    def load_dataset_test(self, batchsize):
        for start_idx in range(0, self.X_test.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield self.X_test[excerpt,:], self.y_test_bin[excerpt,:]

    def train(self, num_epochs=10, batch_size=100):
        print('Start training ...')
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        for epoch in range(num_epochs):
            print('Epoch %2d/%2d: ' % (epoch + 1, num_epochs))

            for batch_id, batch in enumerate(self.load_dataset_train(batch_size)):
                X_train, y_train_bin = batch
                cnn_features, mlp_features = np.hsplit(X_train, [150])

                self.sess.run(self.train_opt, feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.y_labels:y_train_bin, self.keep_prob:0.9})

                train_loss_batch = self.evaluate(mlp_features, cnn_features, y_train_bin)
                train_acc_batch = self.get_accuracy(mlp_features, cnn_features, y_train_bin)

                print('%d: loss = %8.4f, acc = %3.2f%%' % (batch_id, train_loss_batch, train_acc_batch * 100))

    def test(self, batch_size=100):
        print('Start testing ...')

        for batch_id, batch in enumerate(self.load_dataset_test(batch_size)):
            X_test, y_test_bin = batch
            cnn_features, mlp_features = np.hsplit(X_test, [150])

            test_loss_batch = self.evaluate(mlp_features, cnn_features, y_test_bin)
            test_acc_batch = self.get_accuracy(mlp_features, cnn_features, y_test_bin)

            print('%d: loss = %8.4f, acc = %3.2f%%' % (batch_id, test_loss_batch, test_acc_batch * 100))

    def evaluate(self, mlp_features, cnn_features, y):
        return self.loss.eval(feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.keep_prob:1.0, self.y_labels:y}, session=self.sess)

    def get_accuracy(self, mlp_features, cnn_features, y):
        return self.accuracy.eval(feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.keep_prob:1.0, self.y_labels:y}, session=self.sess)

    def get_predicted_labels(self, mlp_features, cnn_features):
        return self.labels.eval(feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.keep_prob:1.0}, session=self.sess)

    def get_hidden_features(self, mlp_features, cnn_features):
        return self.y_raw.eval(feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.keep_prob:1.0}, session=self.sess)

    def save(self):
        model_input_mlp = tf.saved_model.utils.build_tensor_info(self.X_mlp_features)
        model_input_cnn = tf.saved_model.utils.build_tensor_info(self.X_cnn_features)
        model_input_keep_prob = tf.saved_model.utils.build_tensor_info(self.keep_prob)
        model_output_label = tf.saved_model.utils.build_tensor_info(self.labels)
        model_output_y_raw = tf.saved_model.utils.build_tensor_info(self.y_raw)

        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'mlp_features': model_input_mlp, 'cnn_features': model_input_cnn, 'keep_prob': model_input_keep_prob},
                outputs={'label': model_output_label, 'y_raw': model_output_y_raw},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder('./models/' + str(int(time.time())))

        builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.SERVING],
                                signature_def_map={
                                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                        signature_definition
                                })

        builder.save()

    def restore(self):
        self.sess = tf.Session()
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], get_latest_model())

        self.X_mlp_features = tf.get_default_graph().get_tensor_by_name('X_mlp_features:0')
        self.X_cnn_features = tf.get_default_graph().get_tensor_by_name('X_cnn_features:0')
        self.y_labels = tf.get_default_graph().get_tensor_by_name('y_labels:0')
        self.keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        self.loss = tf.get_default_graph().get_tensor_by_name('loss:0')
        self.accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')
        self.train_opt = tf.get_default_graph().get_tensor_by_name('Variable/Adam:0')
        self.labels = tf.get_default_graph().get_tensor_by_name('labels:0')
        self.y_raw = tf.get_default_graph().get_tensor_by_name('y_raw:0')

    def retrain(self, num_epochs=10, batch_size=100):
        print('Start retraining ...')

        for epoch in range(num_epochs):
            print('Epoch %2d/%2d: ' % (epoch + 1, num_epochs))

            for batch_id, batch in enumerate(self.load_dataset_train(batch_size)):
                X_train, y_train_bin = batch
                cnn_features, mlp_features = np.hsplit(X_train, [150])

                self.sess.run(self.train_opt, feed_dict={self.X_mlp_features:mlp_features, self.X_cnn_features:cnn_features, self.y_labels:y_train_bin, self.keep_prob:0.9})

                train_loss_batch = self.evaluate(mlp_features, cnn_features, y_train_bin)
                train_acc_batch = self.get_accuracy(mlp_features, cnn_features, y_train_bin)

                print('%d: loss = %8.4f, acc = %3.2f%%' % (batch_id, train_loss_batch, train_acc_batch * 100))


if __name__ == '__main__':
    nn_instance = NN("./objects.p", labels_length=29)
    nn_instance.build()

    skf = nn_instance.split_train_test(3, 0)

    for train_index, test_index in skf:
        nn_instance.prepare_data(train_index, test_index)
        nn_instance.train()
        nn_instance.test()

    nn_instance.save()
