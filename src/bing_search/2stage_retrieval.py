import models.BM25_retriever as BM25
import argparse
from pathlib import Path
import json
from nltk import word_tokenize
# file = results_dir / 'parsed_bing_results_5_q.json'

       
def retrieve_from_doc(docs, question, args):
    unit2span = {}
    unit2doc_name = {}

    text_units = []
    tokenized_doc = []

    if args.text_unit == "doc":
        tokenized_doc = word_tokenize(" ".join(docs))
        doc_string = " ".join(tokenized_doc)
        text_units.append(tokenized_doc)
    else:
        for doc in docs:
            tokenized_doc = word_tokenize(doc['page_text'])

            for start in range(
                    0, len(tokenized_doc) - args.window_size,
                    args.stride):
                end = start + args.window_size
                unit = tokenized_doc[start:end]
                text_units.append(unit)
                unit2doc_name[' '.join(unit)] = doc['page_name']
                unit2span[' '.join(unit)] = (start, end)

        if not len(text_units):
            return None


    retriever = BM25.BM25Retriever(text_units)

    tokenized_questions = [word_tokenize(question)]
    retrieve_results = []
    units_so_far = set()
    for tq in tokenized_questions:
        units, scores = retriever.get_top_n_doc(tq, num=args.num)
        for u, s in zip(units, scores):
            u = " ".join(u)
            if u not in units_so_far:
                units_so_far.add(u)
                retrieve_results.append((u, s))
    retrieve_results = sorted(retrieve_results, key=lambda x: x[1], reverse=True)[:args.topk_units]

    return retrieve_results, unit2doc_name

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_unit', type=str, default='',
                        help='text unit to use for retrieval')
    parser.add_argument('--window_size', type=int, default=100, help='window size')
    parser.add_argument('--stride', type=int, default=100, help='stride')
    parser.add_argument('--num', type=int, default=10, help='number of retrieved text units per question')
    parser.add_argument('--topk_units', type=int, default=10, help='number of retrieved units')
    parser.add_argument('--rootdir', type=str, default='.', help='path to root directory')

    args = parser.parse_args()


    """
    Take the results from retrieval_results/parsed_results/, and make it concise
    """
    ANSWER_COUNT=10
    answer_source = 'q' #['q', 'q+a', 'q+sum_a']
    # topk = 10
    args.input_file = 'parsed_bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source)

    rootdir = Path(args.rootdir)

    results_dir = rootdir / 'retrieval_results/parsed_results'
    fw = open((rootdir / 'retrieval_results' / '2stage_results' / Path('2stage_bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source))), 'w')


    data = json.load(open(results_dir / args.input_file))

    doc_id = 0
    for inst in data:
        pages = []
        docs = [l for l in inst['pages'] if 'page_text' in l]
        tuple_ = retrieve_from_doc(docs, inst['question'], args)
        if not tuple_:
            print('continue...')
            continue
        else:
            results, unit2doc_name = tuple_

        for l in results:
            # pages.append({'page_text': l[0], 'score':l[1], 'page_name': unit2doc_name[l[0]]})
            pages.append({"title": unit2doc_name[l[0]], "text": l[0], "doc_id": "bing_10_q:%d"%(doc_id)})
            doc_id += 1

        inst['pages'] = pages
        del inst['answers']
        inst['question_id'] = int(inst['id'])
        del inst['id']
        inst['docs'] = pages


    fw.write(json.dumps(data, indent=4))
    fw.close()

