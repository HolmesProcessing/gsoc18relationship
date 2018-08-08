# [GSoC 2018] Holmes Processing: Automated Malware Relationship Mining

*This project is part of [Google Sumer of Code 2018](https://summerofcode.withgoogle.com/projects/#5950627096559616).*

## Overview

The goals of this project are to 

1. implement a decent learning model to predict labels of each malware sample
2. discover relationships between different malware samples
3. visualize relationships in frontend
4. and build an analytic pipeline to integrate the implemented services.

## Prerequisites

- [Closure Compiler](https://developers.google.com/closure/compiler/)
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/)
- [gRPC](https://grpc.io/)
- [Nginx](https://nginx.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Serving](https://www.tensorflow.org/serving/)

## Installation

1. Generate the gRPC client and server interfaces for feed handling and tensorflow serving
```sh
$ cd src

$ cd feedhandling
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. feed_handling.proto
$ cd ..

$ cd tflearning
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tf_learning.proto
$ cd ..
```
2. Generate the gRPC client interface for frontend service
```sh
$ cd frontend
$ mkdir grpc-js

$ protoc -I=../feedhandling/ --js_out=import_style=closure,binary:./grpc-js
      ../feedhandling/feed_handling.proto
$ protoc -I=./grpc-web/third_party/protobuf/src/google/protobuf/ \
      --js_out=import_style=closure,binary:./grpc-js \
      ./grpc-web/third_party/protobuf/src/google/protobuf/any.proto
$ protoc -I=./grpc-web/net/grpc/gateway/protos/ \
      --js_out=import_style=closure,binary:./grpc-js \
      ./grpc-web/net/grpc/gateway/protos/stream_body.proto
$ protoc -I=./grpc-web/net/grpc/gateway/protos/ \
      --js_out=import_style=closure,binary:./grpc-js \
      ./grpc-web/net/grpc/gateway/protos/pair.proto
```
3. Generate gRPC-Web protoc plugin and the client stub service file (feedhandling.grpc.pb.js)
```sh
$ cd grpc-web/javascript/net/grpc/web
$ make

$ cd -
$ protoc -I=. --plugin=protoc-gen-grpc-web=<path to>/protoc-gen-grpc-web \
      --grpc-web_out=out=feedhandling.grpc.pb.js,mode=grpcweb:. \
      ../feedhandling/feed_handling.proto
```
4. Compile all the relevant JS files into one single JS library that can be used in the browser
```sh
$ java \
      -jar <path to>/closure-compiler.jar \
      --js ./grpc-web/javascript \
      --js ./grpc-web/net \
      --js ./grpc-web/third_party/closure-library \
      --js ./grpc-web/third_party/protobuf/js \
      --js ./grpc-js \
      --entry_point=goog:proto.feedhandling.FeedHandlingClient \
      --dependency_mode=STRICT \
      --js_output_file ./grpc-js/compiled.js
```
5. Compile specific modules into Nginx in order for grpc-web requests to be interpreted and proxied to the backend gRPC server
```sh
$ cd grpc-web
$ make package
```

## Usage

### Preprocessing

Programs in `src/preprocessing` are responsible for data preprocessing. Please run in the following order:

```
preprocess_SERVICE_NAME.scala
preprocess_SERVICE_NAME.py
```

After preprocessing, the data will be store in the database for further usage.

Currently there are four supported services: Cuckoo, Objdump, PEinfo, Rich header.

### Neural networks

With the preprocessed data from the previous step, we can use `src/tflearning/NN.py` to train the learning model.

```python
from tflearning.NN import NN

nn_instance = NN(PREPROCESSED_DATA_PATH, labels_length=29)
nn_instance.build()

skf = nn_instance.split_train_test(3, 0)

for train_index, test_index in skf:
    nn_instance.prepare_data(train_index, test_index)
    nn_instance.train()
    nn_instance.test()

nn_instance.save()
```

It is also allowed to retrain the learning model with the following script:

```python
nn_instance = NN(PREPROCESSED_DATA_PATH, label_length=29)
nn_instance.restore()

skf = nn_instance.split_train_test(3, 0)

for train_index, test_index in skf:
    nn_instance.prepare_data(train_index, test_index)
    nn_instance.retrain()
```

### Relationship Discovery

In order to discover the relationships between different malware samples, we build a [KDTree](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) using `src/relationship/FeatureTree.py`.

```sh
$ python relationship/FeatureTree.py
```

### Analytic pipeline

#### Feed handling

```
usage: python fh_server.py [-h] [-v] [-p PORT] [--tfl-addr TFL_ADDR]
                           [--cluster-ip [CLUSTER_IP [CLUSTER_IP ...]]]
                           [--cluster-port CLUSTER_PORT]
                           [--auth-username AUTH_USERNAME]
                           [--auth-password AUTH_PASSWORD] [--offline]

Feed handling server

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -p PORT, --port PORT  Listening port for feed handling server
  --tfl-addr TFL_ADDR   Address of tensorflow learning server
  --cluster-ip [CLUSTER_IP [CLUSTER_IP ...]]
                        IPs of clusters
  --cluster-port CLUSTER_PORT
                        Port of clusters
  --auth-username AUTH_USERNAME
                        Username for clusters' authentication
  --auth-password AUTH_PASSWORD
                        Password for clusters' authentication
  --offline             Offline mode
```

#### Tensorflow serving

```
usage: python tfl_server.py [-h] [-v] [-p PORT] [--fh-addr FH_ADDR]
                            [--model-path MODEL_PATH] [--offline]

Tensorflow learning server

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -p PORT, --port PORT  Listening port for tensorflow learning server
  --fh-addr FH_ADDR     Address of feed handling server
  --model-path MODEL_PATH
                        Location of the learning model
  --offline             Offline mode
```

#### Frontend

Before running Nginx, the absolute path to `src/frontend` should be provided:

```
...

server {
  listen 8888;
  server_name localhost;
  location / {
    root <path-to>/frontend;
    include /etc/nginx/mime.types;
  }

...
```

```sh
$ cp src/frontend/nginx.conf src/frontend/grpc-web/gConnector/conf
$ cd src/frontend/grpc-web/gConnector && ./nginx.sh &
```

Afterwards, view the visualization of relationships in <http://localhost:9090/index.html>.

## Testing

1. Copy and put all the necessary sample data into `src/relationship`

```sh
cp tests/*.p src/relationship
```

2. Run feed handling and Tensorflow learning servers

```sh
$ python fh_server.py -v -p 9090 --tfl-addr localhost:9091 --offline
$ python tfl_server.py -v -p 9091 --fh-addr localhost:9090 --offline
```

3. Run the Nginx service

```sh
$ cp src/frontend/nginx.conf src/frontend/grpc-web/gConnector/conf
$ cd src/frontend/grpc-web/gConnector && ./nginx.sh &
```

4. Play around the relationships of sample data in <http://localhost:9090/index.html>.
