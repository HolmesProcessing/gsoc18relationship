import grpc
import pickle

import feed_handling_pb2
import feed_handling_pb2_grpc


def get_training_data(stub):
    rows = stub.GetTrainingData(feed_handling_pb2.Empty())

    object_list = []

    for r in rows:
        object_list.append(r.SerializeToString())

    f = open('objects.p', 'wb')
    pickle.dump(object_list, f)
    f.close

def run():
    channel = grpc.insecure_channel(SERVER_IP + ':50051')
    stub = feed_handling_pb2_grpc.FeedHandlingStub(channel)
    get_training_data(stub)

if __name__ == '__main__':
    run()
