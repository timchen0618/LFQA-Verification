import json
import argparse
from pathlib import Path
import csv    

def aggregate_factscore(input_files, output_file, nli_file, threshold=0.5):
    out_data = {}
    for f in input_files:
        data = json.load(open(f))
        for key in data.keys():
            if key != 'avg':
                out_data[key] = []
                # print(data[key])
                for l in data[key]:
                    out_data[key].append({'pred': int(l >= threshold)})


    fw = open(output_file, 'w')
    fw.write(json.dumps(out_data, indent=4))
    fw.close()

def aggregate_qafactevalscore(input_files, output_file, annotation_files, nli_files, key, threshold):
    out_data = {}
    for input_file, annotation_file, nli_file in zip(input_files, annotation_files, nli_files):
        assert input_file.stem.split('-')[-1] == annotation_file.stem.split('-')[-1]
        assert nli_file.stem.split('-')[-1].split('_')[-1] == annotation_file.stem.split('-')[-1].split('_')[-1], (nli_file.stem.split('-')[-1], annotation_file.stem.split('-')[-1])

        data = json.load(open(input_file))
        annotations = json.load(open(annotation_file))
        nli_data = json.load(open(nli_file))
        for i, inst in enumerate(data):
            annotation_inst = annotations[i]
            assert inst['question'] == annotation_inst['question']
            scores = inst['positive_ctxs_scores']['answers']
            questionid = annotation_inst['questionid']
            nli_inst = nli_data[questionid]
            pairs = nli_inst['hyp_label_pairs']

            out_data[questionid] = []


            for pair in pairs:
                answer_id = pair['answer_id']
                score_dict = scores[int(answer_id)]
                score = score_dict[key]
                out_data[questionid].append({'pred': int(score >= threshold)})


    fw = open(output_file, 'w')
    fw.write(json.dumps(out_data, indent=4))
    fw.close()


def aggregate_results(args):
    results_prefix = args.results_prefix
    result_dir = Path(args.result_dir)
    all_results = result_dir.glob(f'{results_prefix}*')
    all_results = sorted(all_results, key=lambda x: x.stem.split('_')[-1])
    out_dict = {"Pred Supported": [], "ER": [], "F1": []}
    for f in all_results:
        threshold = f.stem.split('_')[-1]
        reader = csv.reader(open(f), delimiter='\t')
        for row in reader:
            out_dict[row[0]].append([str(threshold)] + row[1:])

    fw = open(result_dir / f'agg_{results_prefix}.tsv', 'w')
    for k, v in out_dict.items():
        fw.write(k + '\n')
        for row in v:
            fw.write('\t'.join(row) + '\n')
        fw.write('\n')



def main(args):
    # command = 'aggregate_factscore'
    # command = 'aggregate_qafactevalscore'
    # command = 'aggregate_results'

    if command == 'aggregate_factscore':
        THRESHOLD = 0.07
        input_files = ['webgpt.json', 'gpt3_wdoc.json', 'gpt3_whudoc.json', 'alpaca_wdoc.json', 'gpt3_003.json', 'alpaca.json']
        # input_files = ['webgpt.json', 'gpt3_wdoc.json', 'gpt3_whudoc.json', 'gpt3_wdoc.json', 'alpaca.json']
        input_files = [Path('FActScore/scores') / f for f in input_files]
        output_file = 'predictions/pred_factscore_%1.2f.json'%(THRESHOLD)
        nli_file = json.load(open(args.input_file))
        # nli_file = json.load(open(nli_file[0]['input_file']))
        aggregate_factscore(input_files, output_file, nli_file, THRESHOLD)
    elif command == 'aggregate_qafactevalscore':
        key = 'lerc_quip'
        # key = 'is_answered'
        THRESHOLD = {'lerc_quip':0.4, 'is_answered':0.9}

        # input_files = ['webgpt.json', 'gpt3_wdoc.json', 'gpt3_whudoc.json', 'alpaca_wdoc.json', 'gpt3_003.json', 'alpaca.json']
        files = ['webgpt.json', 'gpt3_wdoc.json', 'gpt3_whudoc.json', 'alpaca_wdoc.json', 'gpt3_003.json', 'alpaca.json']
        # files = ['webgpt.json']
        # input_files = ['webgpt.json', 'gpt3_wdoc.json', 'gpt3_whudoc.json', 'gpt3_wdoc.json', 'alpaca.json']
        annotation_files = [Path('../Data/github_data') / ('annotations-%s'%f) for f in files]
        input_files = [Path('QAFactEval') / ('qafacteval-%s'%f) for f in files]
        output_file = 'predictions/pred_qafactevalscore_%s_%1.2f.json'%(key, THRESHOLD[key])
        
        nli_files = []
        for f in files:
            if f == 'gpt3_whudoc.json':
                nli_files.append(Path('../Data/nli_data/main-nli-all96_%s'%(f)))
            else:
                nli_files.append(Path('../Data/nli_data/main-nli-all100_%s'%(f)))

        aggregate_qafactevalscore(input_files, output_file, annotation_files, nli_files, key, THRESHOLD[key])
    elif command == 'aggregate_results':
        aggregate_results(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="../Data/nli_data/test_files.json", type=str)
    parser.add_argument("--pred_file", default="pred.json", type=str)
    parser.add_argument("--pred_file_list", default=[], nargs='+', type=str)
    parser.add_argument("--output_file", default="results/results.tsv", type=str)
    parser.add_argument("--results_prefix", default="results_t5", type=str)
    parser.add_argument("--result_dir", default="results", type=str)

    args = parser.parse_args()

    main(args)


# aggregate_results
# python eval.py --results_prefix results_t5 --result_dir results

