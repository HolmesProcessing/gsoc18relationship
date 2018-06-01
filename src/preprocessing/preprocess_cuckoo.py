import ast
import json

cuckoo_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
cuckoo_results = cuckoo_rdd.map(lambda x: (x.sha256, x.service_name, find_api_call_in_cuckoo(x.results), convert_to_list(x.source_tags)))

def find_api_call_in_cuckoo(results):
    api_call_list = []

    try:
        for j in ast.literal_eval(results):
            if j['Subtype'] == 'api_call':
                api_call_list.append(j['Result'])
    except:
        pass

    if len(api_call_list) != 100:
        api_call_list.extend(' ' * (100 - len(api_call_list)))

    return api_call_list

def convert_to_list(source_tags):
    return source_tags[1:len(source_tags) - 1].split(',')

cuckoo_df = cuckoo_results.toDF()
cuckoo_df = cuckoo_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "label")
cuckoo_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_TABLE, keyspace=KEYSPACE).save()
