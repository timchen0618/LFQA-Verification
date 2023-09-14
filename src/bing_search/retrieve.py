import json
import models.raw_evidence_retriever as retriever

PAGE_LIMIT = 50
rootdir = 'retrieval_results/raw_results/'

def retrieve_one_ex(web_retriever, answer_source, ANSWER_COUNT):
    all_pages = []

    if answer_source == 'q':
        query = q
    elif answer_source == 'q+a':
        query = q + '[SEP]' + answer
    elif answer_source == 'q+sum_a':
        raise NotImplementedError
    else:
        raise NotImplementedError

    current_count = ANSWER_COUNT
    offset = 0
    while current_count > 0:
        if current_count > PAGE_LIMIT:
            raw_count = PAGE_LIMIT                    
        else:
            raw_count=current_count

        res = web_retriever.get_results(query, raw_count=raw_count, offset=offset)
        pages_per_retrieve = res['pages_info']
        # print(current_count, len(pages_per_retrieve))
        all_pages += pages_per_retrieve
        current_count -= len(pages_per_retrieve)
        offset += raw_count

    return all_pages



data = {}
for ANSWER_COUNT in [10]:
    web_retriever = retriever.WebRetriever(
        engine='bing',
        answer_count=ANSWER_COUNT,
    )

    for answer_source in ['q']:
        print("doing answer count %d, answer source %s"%(ANSWER_COUNT, answer_source))
        data = json.load(open('../Data/ui_data/ui_data_003.json'))

        json_writer = open(rootdir + 'bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source), 'w')
        all_retrieved_results = []

        for i in range(len(data)):
            inst = {}
            csv_string = ''
            q = data[str(i)]['input']['question']
            answer = ' '.join(data[str(i)]['input']['answers'])
            id_ = str(i)
            csv_string += '%s\t%s\t%s\t'%(id_, q, answer)
            inst['question'] = q
            inst['answers'] =  data[str(i)]['input']['answers']
            inst['id'] = id_

            inst['pages'] = retrieve_one_ex(web_retriever, answer_source, ANSWER_COUNT)            
            # retrieve more
            all_retrieved_results.append(inst)
        
        json_writer.write(json.dumps(all_retrieved_results, indent=4))
        json_writer.close()
