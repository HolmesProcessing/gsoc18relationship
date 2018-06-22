import pickle
import random
import numpy as np
import tensorflow as tf
import feed_handling_pb2

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
    return tf.nn.avg_pool(x, ksize=[1, 150, 1, 1], strides=[1, 150, 1, 1], padding='SAME')

def encode_label(label):
    labels = ['CryptoRansom', 'apt1', 'athena_variant', 'betabot', 'blackshades', 'citadel_krebs', 'darkcomet', 'darkddoser', 'dirtjumper', 'expiro', 'gamarue', 'ghostheart2', 'locker', 'machbot', 'mediyes', 'nitol', 'pushdo', 'shylock', 'simda', 'yoyoddos2']
    return labels.index(label)

class NN:
    def __init__(self, file_path, label_length, learning_rate=0.1):
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
                self.X_cuckoo.append(np.zeros(150))
            else:
                self.X_cuckoo.append(np.array(map(float, reader.features_cuckoo)))

            if not reader.features_objdump:
                self.X_objdump.append(np.zeros(100))
            else:
                self.X_objdump.append(np.array(map(float, reader.features_objdump)))

            if not reader.features_peinfo:
                self.X_peinfo.append(np.zeros(17))
            else:
                self.X_peinfo.append(np.array(map(float, reader.features_peinfo)))

            if not reader.features_richheader:
                self.X_richheader.append(np.zeros(80))
            else:
                self.X_richheader.append(np.array(map(float, reader.features_richheader)))

            self.y.append(encode_label(reader.label))

        self.X_cuckoo = np.array(self.X_cuckoo)
        self.X_objdump = preprocessing.MinMaxScaler().fit_transform(self.X_objdump)
        self.X_peinfo = preprocessing.MinMaxScaler().fit_transform(self.X_peinfo)
        self.X_richheader = preprocessing.MinMaxScaler().fit_transform(self.X_richheader)

        self.X = np.concatenate((self.X_cuckoo, self.X_objdump, self.X_peinfo, self.X_richheader), axis=1)
        self.y = np.array(self.y)

        self.label_length = label_length
        self.W_out = None
        self.b_out = None

        tf.set_random_seed(1337)
        self.build(learning_rate)

    def split_train_test(self, num_splits, random_state):
        skf = StratifiedKFold(n_splits=num_splits, random_state=random_state)
        return skf.split(self.X, self.y)

    def resample(self, random_state):
        ros = RandomOverSampler(random_state=random_state)
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

        self.resample(42)

        num_y = np.size(self.y_train)
        self.y_train_bin = np.zeros((num_y, self.label_length))

        for i in range(num_y):
            self.y_train_bin[i, self.y_train[i]] = 1

    def encode_cnn_features(self, cnn_features, batch_size):
        x_cnn = np.empty((0, 150), dtype=np.float64)
        for i in range(batch_size):
            x = np.zeros((150, 322), dtype=np.float64)
            for j in range(150):
                x[j, int(cnn_features[i, j])] = True
            x = np.transpose(x)
            x_cnn = np.append(x_cnn, x, axis=0)

        return x_cnn.T.reshape(-1, 150, 322)

    def build_mlp(self, features, feature_length):
        W_ff_1 = weight_variable([feature_length, feature_length])
        b_ff_1 = bias_variable([feature_length])
        W_ff_2 = weight_variable([feature_length, feature_length])
        b_ff_2 = bias_variable([feature_length])

        h = tf.nn.relu(tf.matmul(features, W_ff_1) + b_ff_1)
        h_2 = tf.nn.relu(tf.matmul(h, W_ff_2) + b_ff_2)

        return h_2

    def build_cnn(self, features, dict_length):
        W_conv_1 = weight_variable([3, dict_length, 1, 3])
        b_conv_1 = bias_variable([3])
        W_conv_2 = weight_variable([3, dict_length, 3, 6])
        b_conv_2 = bias_variable([6])

        h_conv_1 = tf.nn.relu(conv2d(tf.expand_dims(features, 3), W_conv_1) + b_conv_1)
        h_pool_1 = max_pool_1(h_conv_1)
        h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
        h_pool_2 = max_pool_2(h_conv_2)

        return tf.reshape(h_pool_2, [-1, dict_length * 6])

    def build(self, learning_rate):
        self.train_mlp_features = tf.placeholder(tf.float32, shape=(None, 197), name='mlp_features')
        self.train_cnn_features = tf.placeholder(tf.float32, shape=(None, 150, 322), name='cnn_features')
        self.train_labels = tf.placeholder(tf.int8, shape=(None, self.label_length))

        NN_mlp = self.build_mlp(self.train_mlp_features, 197)
        NN_cnn = self.build_cnn(self.train_cnn_features, 322)
        h_concat = tf.concat([NN_mlp, NN_cnn], 1)

        if not self.W_out:
            self.W_out = weight_variable([2129, self.label_length])

        if not self.b_out:
            self.b_out = bias_variable([self.label_length])

        y_raw = tf.matmul(h_concat, self.W_out) + self.b_out
        self.y_out = tf.nn.softmax(y_raw)
        self.label = tf.argmax(self.y_out, 1, name='get_label')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_out, labels=self.train_labels))

        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(self.label, tf.argmax(self.train_labels, 1))
        self.correct_predictions = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.init_op = tf.global_variables_initializer()

    def train(self, num_epochs=200, batch_size=100):
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        N = self.X_train.shape[0]
        for epoch in range(num_epochs):
            print('Epoch %2d/%2d: ' % (epoch + 1, num_epochs))

            index = [i for i in range(N)]
            batch_id = 0
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop() for i in range(min(batch_size, index_size))]

                cnn_features, mlp_features = np.hsplit(self.X_train[batch_index,:], [150])
                cnn_features_bin = self.encode_cnn_features(cnn_features, batch_size)

                self.sess.run(self.train_opt, feed_dict={self.train_mlp_features:mlp_features, self.train_cnn_features:cnn_features_bin, self.train_labels:self.y_train_bin[batch_index]})
                train_loss_batch = self.loss.eval(feed_dict={self.train_mlp_features:mlp_features, self.train_cnn_features:cnn_features_bin, self.train_labels:self.y_train_bin[batch_index]}, session=self.sess)

                print("{0}: {1}".format(batch_id, train_loss_batch))
                batch_id += 1

            cnn_features, mlp_features = np.hsplit(self.X_train[batch_index,:], [150])
            train_loss = self.evaluate(mlp_features, cnn_features_bin, self.y_train_bin)
            train_acc = self.get_accuracy(mlp_features, cnn_features_bin, self.y_train_bin)
            print(' loss = %8.4f, acc = %3.2f%%' % (train_loss, train_acc * 100))

    def evaluate(self, mlp_features, cnn_features, y):
        return self.loss.eval(feed_dict={self.train_mlp_features:mlp_features, self.train_cnn_features:cnn_features, self.train_labels:y}, session=self.sess)

    def get_accuracy(self, mlp_features, cnn_features, y):
        return self.correct_predictions.eval(feed_dict={self.train_mlp_features:mlp_features, self.train_cnn_features:cnn_features, self.train_labels:y}, session=self.sess) / y.shape[0]

    def save(self):
        model_input_mlp = tf.saved_model.utils.build_tensor_info(self.train_mlp_features)
        model_input_cnn = tf.saved_model.utils.build_tensor_info(self.train_cnn_features)
        model_output = tf.saved_model.utils.build_tensor_info(self.label)

        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'mlp_features': model_input_mlp, 'cnn_features': model_input_cnn},
                outputs={'get_label': model_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder('./models/1')

        builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.SERVING],
                                signature_def_map={
                                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                        signature_definition
                                })

        builder.save()


if __name__ == '__main__':
    nn_instance = NN(FILE_LOCATION, feature_length=FEATURE_LENGTH, label_length=LABEL_LENGTH)

    skf = nn_instance.split_train_test(3, 0)

    for train_index, test_index in skf:
        nn_instance.prepare_data(train_index, test_index)
        nn_instance.train()

    nn_instance.save()
