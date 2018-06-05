import ast
import json

cuckoo_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
cuckoo_results = cuckoo_rdd.map(lambda x: (x.sha256, x.service_name, find_api_call_in_cuckoo(x.results), convert_to_label(x.source_tags)) if 'benign' not in x.source_tags else None).filter(bool)

def find_api_call_in_cuckoo(results):
    api_call_list = []

    try:
        for j in ast.literal_eval(results):
            if j['Subtype'] == 'api_call':
                api_call_list.append(j['Result'])
    except:
        pass

    if len(api_call_list) != 100:
        api_call_list.extend(['nop'] * (100 - len(api_call_list)))

    return api_call_list

def convert_to_label(source_tags):
    labels = source_tags[1:len(source_tags) - 1].split(',')
    return labels[1 - labels.index('malicious')]

cuckoo_df = cuckoo_results.toDF()
cuckoo_df = cuckoo_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "label")
cuckoo_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_TABLE, keyspace=KEYSPACE).save()
