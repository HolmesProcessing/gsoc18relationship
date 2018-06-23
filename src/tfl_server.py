from concurrent import futures
import time
import grpc

from feedhandling import feed_handling_pb2
from feedhandling import feed_handling_pb2_grpc
from tflearning import tf_learning_pb2
from tflearning import tf_learning_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class TFLearningServicer(tf_learning_pb2_grpc.TFLearningServicer):
    def __init__(self):
        pass

    def PredictLabel(self, request, context):
        pass

    def GetRelationships(self, request, context):
        pass

    def TrainModel(self, request, context):
        pass

    def Echo(self, request, context):
        return tf_learning_pb2.Foo(msg=request.msg)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tf_learning_pb2_grpc.add_TFLearningServicer_to_server(TFLearningServicer(), server)
    server.add_insecure_port('[::]:9091')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
