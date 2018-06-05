import grpc
import pickle

import feed_handling_pb2
import feed_handling_pb2_grpc

def save_to_file(service_name, fl_list):
    f = open(service_name + '_features_and_labels.p', 'wb')

    pickle.dump(fl_list, f)

    f.close()

def get_training_data(stub):
    rows = stub.GetTrainingData(feed_handling_pb2.Empty())

    cuckoo_fl_list = []
    objdump_fl_list = []
    peinfo_fl_list = []
    richheader_fl_list = []

    for r in rows:
        locals()[r.service_name.lower() + '_fl_list'].append(r.SerializeToString())

    save_to_file('cuckoo', cuckoo_fl_list)
    save_to_file('objdump', objdump_fl_list)
    save_to_file('peinfo', peinfo_fl_list)
    save_to_file('richheader', richheader_fl_list)

def run():
    channel = grpc.insecure_channel(SERVER_IP + ':50051')
    stub = feed_handling_pb2_grpc.FeedHandlingStub(channel)
    get_training_data(stub)

if __name__ == '__main__':
    run()
