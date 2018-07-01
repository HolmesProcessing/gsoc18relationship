import sys
sys.path.append('../')

import pickle
import numpy as np
import tensorflow as tf
import feedhandling.feed_handling_pb2 as feed_handling_pb2

from sklearn import preprocessing
from sklearn.neighbors import KDTree
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

class FeatureTree:
    def __init__(self, file_path):
        self.X_cuckoo = []
        self.X_objdump = []
        self.X_peinfo = []
        self.X_richheader = []
        self.sha256 = []

        objects = pickle.load(open(file_path, 'rb'))

        reader = feed_handling_pb2.TrainingData()
        for o in objects:
            reader.ParseFromString(o)

            if not reader.features_cuckoo:
                self.X_cuckoo.append(np.zeros(150))
            else:
                self.X_cuckoo.append(np.array(map(int, reader.features_cuckoo)))

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

            self.sha256.append(reader.sha256)

        self.cnn_features = np.array(self.X_cuckoo).astype(np.int32)

        self.X_objdump = preprocessing.MinMaxScaler().fit_transform(self.X_objdump)
        self.X_peinfo = preprocessing.MinMaxScaler().fit_transform(self.X_peinfo)
        self.X_richheader = preprocessing.MinMaxScaler().fit_transform(self.X_richheader)
        self.mlp_features = np.concatenate((self.X_objdump, self.X_peinfo, self.X_richheader), axis=1).astype(np.float32)

        self.hidden_features = np.empty([1, 257])

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

            self.hidden_features = np.concatenate((self.hidden_features, tensor_util.MakeNdarray(response.outputs['h_concat'])), axis=0)

            chunk_id += 1

        self.hidden_features = np.delete(self.hidden_features, 1, 0)
        print('Received all features!\n')

    def build_and_save_feature_tree(self):
        print('Start building and saving the feature tree ...\n')
        self.tree = KDTree(self.hidden_features, leaf_size=257)

        tree_f = open('ftree.p', 'wb')
        sha256_f = open('sha256.p', 'wb')
        pickle.dump(self.tree, tree_f)
        pickle.dump(self.sha256, sha256_f)
        tree_f.close()
        sha256_f.close()

        print('Everything is done and goes well! :)\n')

def main(_):
    ft = FeatureTree('./objects.p')

    ft.get_hidden_features(FLAGS.server)
    ft.build_and_save_feature_tree()

if __name__ == '__main__':
    tf.app.run()

