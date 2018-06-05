import json

richheader_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
richheader_results = richheader_rdd.map(lambda x: (x.sha256, x.service_name, find_compid_in_richheader(x.results), convert_to_label(x.source_tags)) if 'benign' not in x.source_tags else None).filter(bool)

def find_compid_in_richheader(results):
    compid_list = []

    richheader = json.loads(results)

    if 'cmpids' not in richheader:
        return [(0,0,0)] * 20

    for compid in richheader['cmpids']:
        compid_list.append((compid['mcv'], compid['pid'], compid['cnt']))

    compid_list += [(0,0,0)] * (20 - len(compid_list))

    return compid_list

def convert_to_label(source_tags):
    labels = source_tags[1:len(source_tags) - 1].split(',')
    return labels[1 - labels.index('malicious')]

richheader_df = richheader_results.toDF()
richheader_df = richheader_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "label")
richheader_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_TABLE, keyspace=KEYSPACE).save()
