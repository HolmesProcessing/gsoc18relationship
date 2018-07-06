from concurrent import futures
import time
import grpc
import argparse

from feedhandling import feed_handling_pb2
from feedhandling import feed_handling_pb2_grpc
from tflearning import tf_learning_pb2
from tflearning import tf_learning_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def get_training_data_from_storage():
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider

    auth_provider = PlainTextAuthProvider(username=USERNAME, password=PASSWORD)
    cluster = Cluster(LIST_OF_CLUSTERS, port=PORT, auth_provider=auth_provider)
    session = cluster.connect()
    session.set_keyspace(KEYSPACE)

    return session.execute('SELECT * FROM ' + OBJECTS_TABLE)

class FeedHandlingServicer(feed_handling_pb2_grpc.FeedHandlingServicer):
    def __init__(self, verbose):
        self.verbose = verbose

    def QueryRelationship(self, request, context):
        pass

    def SendMalwareSample(self, request, context):
        pass

    def InitiateTraining(self, request, context):
        if self.verbose:
            print('[Request] InitiateTraining()')
            print('[Info] Initiate training the learning model')

        channel = grpc.insecure_channel('localhost:9091')
        stub = tf_learning_pb2_grpc.TFLearningStub(channel)
        stub.TrainModel(tf_learning_pb2.Empty())

        if self.verbose:
            print('[Info] Training done!')

        return feed_handling_pb2.Empty()

    def GetTrainingData(self, request, context):
        if self.verbose:
            print('[Request] GetTrainingData()')
            print('[Info] Start fetching training data from storage')

        rows = get_training_data_from_storage()

        if self.verbose:
            print('[Info] Training data fetched!')
            print('[Info] Start sending training data')

        for r in rows:
            yield feed_handling_pb2.TrainingData(sha256=r.sha256, features_cuckoo=r.features_cuckoo, features_objdump=r.features_objdump, features_peinfo=r.features_peinfo, features_richheader=r.features_richheader, label=r.labels)

        if self.verbose:
            print('[Info] Training data sent!')

    def SendRelationship(self, request, context):
        pass

    def Echo(self, request, context):
        if self.verbose:
            print('[Request] Echo(%s)' % request.msg)

        return feed_handling_pb2.Foo(msg=request.msg)

def serve(verbose):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    feed_handling_pb2_grpc.add_FeedHandlingServicer_to_server(FeedHandlingServicer(verbose), server)
    server.add_insecure_port('[::]:9090')
    server.start()

    if verbose:
        print('[Info] Feed handling server init')

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python fh_server.py', description='Feed handling server')
    parser.add_argument('-v', '--verbose', help='Verbose mode', action='store_true')
    args = parser.parse_args()

    serve(args.verbose)
