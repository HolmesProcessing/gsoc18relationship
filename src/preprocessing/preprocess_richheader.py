import json

richheader_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
richheader_results = richheader_rdd.map(lambda x: (x.sha256, find_compid_in_richheader(x.results)))

def find_compid_in_richheader(results):
    compid_list = []

    richheader = json.loads(results)

    if 'cmpids' not in richheader:
        return [(0,0,0)] * 10000

    for compid in richheader['cmpids']:
        compid_list.append((compid['mcv'], compid['pid'], compid['cnt']))

    compid_list += [(0,0,0)] * (10000 - len(compid_list))

    return compid_list
