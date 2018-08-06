import json
import time

with open(X86OPCODE) as f:
    opcodes = f.read().splitlines()

objdump_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
objdump_results = objdump_rdd.map(lambda x: (x.sha256, x.service_name, find_op_in_objdump(x.results), convert_to_labels(x.source_tags), int(time.time())) if 'benign' not in x.source_tags else None).filter(bool)
objdump_objects = objdump_rdd.map(lambda x: (x.sha256, find_op_in_objdump(x.results), convert_to_labels(x.source_tags), int(time.time())) if 'benign' not in x.source_tags else None).filter(bool)

def find_op_in_objdump(results):
    op_list = []

    sections = json.loads(results)['sections']

    for k, v in sections.items():
        try:
            for b in v['blocks']:
                try:
                    for op in b['opcodes']:
                        op_list.append(opcodes.index(op))
                except:
                    pass
        except:
            pass

    op_list += [0.0] * (100 - len(op_list))

    return op_list

def convert_to_labels(source_tags):
    labels = source_tags[1:len(source_tags) - 1].split(',')
    labels.remove('malicious')
    return labels

objdump_df = objdump_results.toDF()
objdump_df = objdump_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "labels").withColumnRenamed("_5", "timestamp")
objdump_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_RESULTS, keyspace=KEYSPACE).save()

objdump_df = objdump_objects.toDF()
objdump_df = objdump_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "features_objdump").withColumnRenamed("_3", "labels").withColumnRenamed("_4", "timestamp")
objdump_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_OBJECTS, keyspace=KEYSPACE).save()

