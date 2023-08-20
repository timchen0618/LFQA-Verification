import json
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
from collections import Counter

def acc(pred, label):
    assert len(pred) == len(label)
    return sum([p==l for (p, l) in zip(pred, label)])/float(len(label))

def convert_label_count_to_binary_label(label_count, label_space):
    binary_label_count = {0: 0, 1: 0}
    for k, v in label_count.items():
        binary_label_count[label_space[k]] += v

    return binary_label_count
    
def ensemble(pred_file_list):
    pred_dict = {}
    keys = json.load(open(pred_file_list[0])).keys()

    for f in pred_file_list:
        dict_ = json.load(open(f))
        for key in keys:
            if key not in pred_dict:
                pred_dict[key] = dict_[key]
                for i in range(len(pred_dict[key])):
                    pred_dict[key][i]['pred'] = [dict_[key][i]['pred']]
            else:
                assert len(pred_dict[key]) == len(dict_[key])   
                for i in range(len(pred_dict[key])):
                    pred_dict[key][i]['pred'].append(dict_[key][i]['pred'])

    # merge predictions
    for key in keys:
        for i in range(len(pred_dict[key])):
            if len(pred_dict[key][i]['pred']) == 2:
                pred_dict[key][i]['pred'] = pred_dict[key][i]['pred'][0] * pred_dict[key][i]['pred'][1]
            elif len(pred_dict[key][i]['pred']) >= 3:
                c = Counter(pred_dict[key][i]['pred'])
                value, count = c.most_common()[0]
                pred_dict[key][i]['pred'] = value

    return pred_dict


def main(args):
    f1_across_files = []
    er_across_files = []
    predicted_supported_across_files = []
    acc_across_files = []

    if args.ensemble:
        pred_dict = ensemble(args.pred_file_list)
    else:
        pred_dict = json.load(open(args.pred_file))

    all_test_files = json.load(open(args.input_file))
    for f_in in all_test_files:
        label_count = {'partially':0, 'supported':0, 'not_supported':0}
        label_space = {'partially':0, 'supported':1, 'not_supported':0}
        data = json.load(open(f_in['input_file']))
        keys = data.keys()
        outputs = []
        labels = []
        ones = []

        for key in tqdm(keys):
            hyp_label_pairs = data[key]['hyp_label_pairs']
            preds = pred_dict[key]
            for pair in hyp_label_pairs:
                labels.append(label_space[pair['label']])
                label_count[pair['label']] += 1                       
            for pair in preds:
                outputs.append(pair['pred'])
                ones.append(int(pair['pred']))

        acc_ = 100*acc(outputs, labels)
        print(labels)
        print(outputs)
        f1_ =  100*f1_score(labels, outputs, pos_label=f_in['positive_label'])
        predicted_supported_ = 100*sum(ones)/float(len(ones))

        binary_label_count = convert_label_count_to_binary_label(label_count, label_space)
        true_positive_rate = 100*binary_label_count[1] / float(binary_label_count[1] + binary_label_count[0])
        er_ = abs(true_positive_rate - predicted_supported_)

        f1_across_files.append(f1_)
        er_across_files.append(er_)
        predicted_supported_across_files.append(predicted_supported_)
        acc_across_files.append(acc_)

    print(args.output_file)
    with open(args.output_file, 'w') as f:
        f.write('F1\t')
        for l in f1_across_files:
            f.write('%2.2f\t'%l)
        f.write('\nER\t')
        for l in er_across_files:
            f.write('%2.2f\t'%l)
        f.write('\nACC\t')
        for l in acc_across_files:
            f.write('%2.2f\t'%l)
        f.write('\nPred Supported\t')
        for l in predicted_supported_across_files:
            f.write('%2.2f\t'%l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="input_file.json", type=str)
    parser.add_argument("--pred_file", default="pred.json", type=str)
    parser.add_argument("--ensemble", action='store_true')
    parser.add_argument("--pred_file_list", default=[], nargs='+', type=str)
    parser.add_argument("--output_file", default="results/results.tsv", type=str)

    args = parser.parse_args()

    main(args)

# only use this when you have the predictions and the data, but did not compute the supported score

# eval 
# python eval.py --pred_file predictions/pred.json --output_file results/results.json 

# eval ensembled models 
# python eval.py --pred_file  predictions/pred_a.json predictions/pred_b.json --output_file results/results.json 

