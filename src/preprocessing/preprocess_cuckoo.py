import ast
import json

cuckoo_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
cuckoo_results = cuckoo_rdd.map(lambda x: (x.sha256, find_api_call_in_cuckoo(x.results)))

def find_api_call_in_cuckoo(results):
    api_call_list = []

    for j in ast.literal_eval(results):
        if j['Subtype'] == 'api_call':
            api_call_list.append(j['Result'])

    return api_call_list

