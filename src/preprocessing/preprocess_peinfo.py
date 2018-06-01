import json

peinfo_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
peinfo_results = peinfo_rdd.map(lambda x: (x.sha256, x.service_name, find_val_in_peinfo(x.results), convert_to_list(x.source_tags)))

def find_val_in_peinfo(results):
    val_list = [0.0] * 16

    peinfo = json.loads(results)

    for s in peinfo["Sections"]:
        if s["Name"]["Value"] == ".text\\x00\\x00\\x00":
            i = 0
        elif s["Name"]["Value"] == ".data\\x00\\x00\\x00":
            i = 1
        elif s["Name"]["Value"] == ".rsrc\\x00\\x00\\x00":
            i = 2
        elif s["Name"]["Value"] == ".rdata\\x00\\x00":
            i = 3
        else:
            continue

        val_list[i * 4] = s["entrophy"]
        val_list[i * 4 + 1] = s["VirtualAddress"]["Value"] + 0.0
        val_list[i * 4 + 2] = s["Misc_VirtualSize"]["Value"] + 0.0
        val_list[i * 4 + 3] = s["SizeOfRawData"]["Value"] + 0.0

    val_list.append(int(peinfo["HEADERS"]["FILE_HEADER"]["TimeDateStamp"]["Value"].split(" ")[0], 16) + 0.0)

    return val_list

def convert_to_list(source_tags):
    return source_tags[1:len(source_tags) - 1].split(',')

peinfo_df = peinfo_results.toDF()
peinfo_df = peinfo_df.withColumnRenamed("_1", "sha256").withColumnRenamed("_2", "service_name").withColumnRenamed("_3", "features").withColumnRenamed("_4", "label")
peinfo_df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=PREPROCESSING_TABLE, keyspace=KEYSPACE).save()
