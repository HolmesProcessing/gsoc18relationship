from concurrent import futures
import time
import grpc

import feed_handling_pb2
import feed_handling_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class FeedHandlingServicer(feed_handling_pb2_grpc.FeedHandlingServicer):
    def __init__(self):
        pass

    def GetTrainingData(self, request, context):
        from cassandra.cluster import Cluster
        from cassandra.auth import PlainTextAuthProvider

        auth_provider = PlainTextAuthProvider(username=USERNAME, password=PASSWORD)
        cluster = Cluster(LIST_OF_CLUSTERS, port=PORT, auth_provider=auth_provider)
        session = cluster.connect()
        session.set_keyspace(KEYSPACE)

        rows = session.execute('SELECT * FROM ' + FEATURE_TABLE)
        for r in rows:
            yield feed_handling_pb2.TrainingData(sha256=r.sha256, service_name=r.service_name, features=r.features, labels=r.labels)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    feed_handling_pb2_grpc.add_FeedHandlingServicer_to_server(FeedHandlingServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()

