import json

def write_json(filename, data):
    fw = open(filename, 'w')
    fw.write(json.dumps(data, indent=4))
    fw.close()

for ANSWER_COUNT in [10]:
    for answer_source in ['q']:
        input_file = 'retrieval_results/raw_results/bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source)
        rootOutputDir = 'retrieval_results/urls/'
        out_data = []
        for k, inst in enumerate(json.load(open(input_file))):
            out_inst = []
            for p in inst['pages']:
                out_inst.append(p['page_url'])

            out_data.append(out_inst)
        write_json(rootOutputDir + 'urls_%d_%s.json'%(ANSWER_COUNT, answer_source), out_data)

