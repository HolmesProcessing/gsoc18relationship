import json

with open(X86OPCODE) as f:
    opcodes = f.read().splitlines()

objdump_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
objdump_results = objdump_rdd.map(lambda x: (x.sha256, x.service_name, find_op_in_objdump(x.results), convert_to_list(x.source_tags)))

def find_op_in_objdump(results):
    op_list = []

    sections = json.loads(results)['sections']

    for k, v in sections.items():
        try:
            for b in v['blocks']:
                try:
                    for op in b['opcodes']:
                        op_list.append(op)
                except:
                    pass
        except:
            pass

    op_list += [0.0] * (100 - len(op_list))

    return op_list

def convert_to_list(source_tags):
    return source_tags[1:len(source_tags) - 1].split(',')

objdump_df = objdump_results.toDF()
objdump_df = objdump_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "label")
objdump_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_TABLE, keyspace=KEYSPACE).save()
