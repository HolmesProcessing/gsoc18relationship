import sys
sys.path.append('../')

import pickle
import numpy as np
import tensorflow as tf
import feedhandling.feed_handling_pb2 as feed_handling_pb2
import time

from sklearn import preprocessing
from sklearn.neighbors import KDTree
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util
from collections import Counter

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

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

    if len(classifications) == 1 and switcher.get(label[classifications[0]]):
        classifications.append(switcher.get(label[classifications[0]]))

    return classifications

class FeatureTree:
    def __init__(self, file_path):
        self.file_path = file_path

    def prepare_data(self):
        self.sha256 = []
        self.labels = []

        X_cuckoo = []
        X_objdump = []
        X_peinfo = []
        X_richheader = []

        objects = pickle.load(open(self.file_path, 'rb'))

        reader = feed_handling_pb2.TrainingData()
        for o in objects:
            reader.ParseFromString(o)

            try:
                self.labels.append(encode_label(reader.labels))
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

            self.sha256.append(reader.sha256)

        self.cnn_features = np.array(X_cuckoo).astype(np.int32)

        X_objdump = preprocessing.MinMaxScaler().fit_transform(X_objdump)
        X_peinfo = preprocessing.MinMaxScaler().fit_transform(X_peinfo)
        X_richheader = preprocessing.MinMaxScaler().fit_transform(X_richheader)
        self.mlp_features = np.concatenate((X_objdump, X_peinfo, X_richheader), axis=1).astype(np.float32)

        self.labels = np.array(self.labels)
        self.hidden_features = np.empty([1, 29])
        self.predicted_labels = np.empty([1, 29])

    def get_hidden_features(self, hostport, chunk_size=1000):
        print('Start retrieving hidden features ...\n')
        host, port = hostport.split(':')
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
                                 
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'malware'
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        N = len(self.sha256)
        index = [i for i in range(N)]
        chunk_id = 0
        while len(index) > 0:
            print('Handling chunk %d' % (chunk_id))
            index_size = len(index)
            chunk_index = [index.pop() for i in range(min(chunk_size, index_size))]

            request.inputs['mlp_features'].CopyFrom(tf.contrib.util.make_tensor_proto(self.mlp_features[chunk_index,:], shape=self.mlp_features[chunk_index,:].shape))
            request.inputs['cnn_features'].CopyFrom(tf.contrib.util.make_tensor_proto(self.cnn_features[chunk_index,:], shape=self.cnn_features[chunk_index,:].shape))
            request.inputs['keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0))

            response = stub.Predict(request, 5.0)

            self.hidden_features = np.concatenate((self.hidden_features, tensor_util.MakeNdarray(response.outputs['y_raw'])), axis=0)
            self.predicted_labels = np.concatenate((self.predicted_labels, tensor_util.MakeNdarray(response.outputs['label'])), axis=0)

            chunk_id += 1

        self.hidden_features = np.delete(self.hidden_features, 0, 0)
        self.predicted_labels = np.delete(self.predicted_labels, 0, 0)
        self.predicted_labels = [self.decode_one_hot(l) for l in self.predicted_labels]
        print('Received all features!\n')

    def decode_one_hot(self, labels):
        decoded_labels = []
        for i in range(len(labels)):
            if labels[i] == 1:
                decoded_labels.append(i)

        return decoded_labels

    def build_and_save_feature_tree(self):
        print('Start building and saving the feature tree ...\n')
        self.tree = KDTree(self.hidden_features, leaf_size=29)

        tree_f = open('ftree.p', 'wb')
        sha256_f = open('sha256.p', 'wb')
        hf_f = open('hf.p', 'wb')
        labels_f = open('labels.p', 'wb')
        predicted_labels_f = open('predicted_labels.p', 'wb')

        pickle.dump(self.tree, tree_f)
        pickle.dump(self.sha256, sha256_f)
        pickle.dump(self.hidden_features, hf_f)
        pickle.dump(self.labels, labels_f)
        pickle.dump(self.predicted_labels, predicted_labels_f)

        tree_f.close()
        sha256_f.close()
        hf_f.close()
        labels_f.close()
        predicted_labels_f.close()

        print('Everything is done and goes well! :)\n')

    def evaluate(self):
        self.tree = pickle.load(open('ftree.p', 'rb'))
        self.sha256 = pickle.load(open('sha256.p', 'rb'))
        self.hidden_features = pickle.load(open('hf.p', 'rb'))
        self.labels = pickle.load(open('labels.p', 'rb'))

        for j in range(len(self.labels)):
            dist, ind = self.tree.query(self.hidden_features[j,:].reshape(1, -1), k=100)

            matched = []
            for i in range(len(ind[0])):
                if set(self.labels[j]) == set(self.labels[ind[0][i]]):
                    matched.append(i)

            print(self.labels[j])
            print(matched)

def main(_):
    ft = FeatureTree('./objects.p')

    ft.prepare_data()
    ft.get_hidden_features(FLAGS.server)
    ft.build_and_save_feature_tree()

    ft.evaluate()

if __name__ == '__main__':
    tf.app.run()

