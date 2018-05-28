import json

peinfo_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
peinfo_results = peinfo_rdd.map(lambda x: (x.sha256, find_val_in_peinfo(x.results)))

def find_val_in_peinfo(results):
    val_list = [0] * 16

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
        val_list[i * 4 + 1] = s["VirtualAddress"]["Value"]
        val_list[i * 4 + 2] = s["Misc_VirtualSize"]["Value"]
        val_list[i * 4 + 3] = s["SizeOfRawData"]["Value"]

    val_list.append(peinfo["HEADERS"]["FILE_HEADER"]["TimeDateStamp"]["Value"].split(" ")[0])

    return val_list
