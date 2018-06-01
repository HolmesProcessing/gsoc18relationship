import grpc
import pickle

import feed_handling_pb2
import feed_handling_pb2_grpc

def save_to_file(service_name, feature_list, label_list):
    f = open(service_name + '_features.p', 'wb')
    l = open(service_name + '_labels.p', 'wb')

    pickle.dump(feature_list, f)
    pickle.dump(label_list, l)

    f.close()
    l.close()

def get_training_data(stub):
    rows = stub.GetTrainingData(feed_handling_pb2.Empty())

    cuckoo_feature_list = []
    objdump_feature_list = []
    peinfo_feature_list = []
    richheader_feature_list = []

    cuckoo_label_list = []
    objdump_label_list = []
    peinfo_label_list = []
    richheader_label_list = []

    for r in rows:
        locals()[r.service_name.lower() + '_feature_list'].append(r.features)
        locals()[r.service_name.lower() + '_label_list'].append(r.labels)

    save_to_file('cuckoo', cuckoo_feature_list, cuckoo_label_list)
    save_to_file('objdump', objdump_feature_list, objdump_label_list)
    save_to_file('peinfo', peinfo_feature_list, peinfo_label_list)
    save_to_file('richheader', richheader_feature_list, richheader_label_list)

def run():
    channel = grpc.insecure_channel(SERVER_IP + ':50051')
    stub = feed_handling_pb2_grpc.FeedHandlingStub(channel)
    get_training_data(stub)

if __name__ == '__main__':
    run()

