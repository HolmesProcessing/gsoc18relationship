import json

with open(X86OPCODE) as f:
    opcodes = f.read().splitlines()

objdump_rdd = sqlContext.read.parquet(DF_LOCATION).rdd
objdump_results = objdump_rdd.map(lambda x: (x.sha256, find_op_in_objdump(x.results)))

def find_op_in_objdump(results):
    op_list = []

    sections = json.loads(results)['sections']

    for k, v in sections.items():
        if v['blocks']:
            for b in v['blocks']:
                if b['opcodes']:
                    for op in b['opcodes']:
                        op_list.append(opcodes.index(op) + 1.0)

    op_list += [0.0] * (10000 - len(op_list))

    return op_list
