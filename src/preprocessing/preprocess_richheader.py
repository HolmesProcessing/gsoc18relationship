import json
import time

richheader_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
richheader_results = richheader_rdd.map(lambda x: (x.sha256, x.service_name, find_compid_in_richheader(x.results), convert_to_labels(x.source_tags), int(time.time())) if 'benign' not in x.source_tags else None).filter(bool)
richheader_objects = richheader_rdd.map(lambda x: (x.sha256, find_compid_in_richheader(x.results), convert_to_labels(x.source_tags), int(time.time())) if 'benign' not in x.source_tags else None).filter(bool)

def find_compid_in_richheader(results):
    compid_list = []

    richheader = json.loads(results)

    if 'cmpids' not in richheader:
        return [0] * 80

    for compid in richheader['cmpids']:
        compid_list.extend([compid['mcv'], compid['pid'], compid['cnt']])

    compid_list += [0] * (80 - len(compid_list))

    return compid_list

def convert_to_labels(source_tags):
    labels = source_tags[1:len(source_tags) - 1].split(',')
    labels.remove('malicious')
    return labels

richheader_df = richheader_results.toDF()
richheader_df = richheader_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "labels").withColumnRenamed("_5", "timestamp")
richheader_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_RESULTS, keyspace=KEYSPACE).save()

richheader_df = richheader_objects.toDF()
richheader_df = richheader_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "features_richheader").withColumnRenamed("_3", "labels").withColumnRenamed("_4", "timestamp")
richheader_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_OBJECTS, keyspace=KEYSPACE).save()

