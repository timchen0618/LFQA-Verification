import logging
import argparse
import json
import csv
import torch
from model_summac import SummaCZS, SummaCConv
from tqdm import tqdm
import random

from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from test_on_docNLI_RoBERTa import RobertaForSequenceClassification, convert_examples_to_features, evaluation
from load_data_docnli import load_NLIdataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_map = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0, "contradiction_idx": 2},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
}


def load_ckpt(model, device, args):
    ckpt = torch.load(args.initialize_model_from_checkpoint, map_location=device)
    model.load_state_dict(ckpt)


def _label_agreement(input_file):
    """
        Store the label agreements as the sam format of webgpt data
        "0": [(True, "partially"), (False, "supported"), ...]
        "1": ...
        ...

        first item stands for whether agree, second is the annotation of first annotator

        Purpose of this function: check whether label agrees -> only include examples where labels agree. 
    """
    agreement_data = {}
    with open(input_file) as ifile:
        reader = csv.reader(ifile, delimiter=",")
        for row in reader:
            if row[0] == 'questionid' or row[0] == '':
                continue
            if str(row[0]) not in agreement_data:
                agreement_data[str(row[0])] = [(False, None) for _ in range(20)]
            
            label_1, label_2 = row[2], row[3]
            agreement_data[str(row[0])][int(row[1])] = (label_1 == label_2, label_1)

    return agreement_data

def acc(pred, label):
    assert len(pred) == len(label)
    return sum([p==l for (p, l) in zip(pred, label)])/float(len(label))


def random_baseline(data, label_count, label_space, keys, positive_label=0):
    accs = []
    f1s = []
    perc_ones = []

    for key in tqdm(keys):
        hyp_label_pairs = data[key]['hyp_label_pairs']
        for pair in hyp_label_pairs:
            label_count[pair['label']] += 1                       
    print(label_count)
    print('Percentage of each label:','supported', '%2.2f'%(100*label_count['supported']/float(sum(label_count.values()))), 'partially', '%2.2f'%(100*label_count['partially']/float(sum(label_count.values()))),'not_supported', '%2.2f'%(100*label_count['not_supported']/float(sum(label_count.values()))))

    positive_rate = label_count['supported']/float(sum(label_count.values()))


    for _ in range(10):
        outputs = []
        labels = []
        ones = []
        for key in keys:
            hyp_label_pairs = data[key]['hyp_label_pairs']
            for pair in hyp_label_pairs:
                labels.append(label_space[pair['label']])
                pred = 1 if random.random() < positive_rate else 0
                outputs.append(int(pred))
                ones.append(int(pred) == 1)

        acc_ = acc(outputs, labels)
        f1 =  f1_score(labels, outputs, pos_label=positive_label)
        perc_one = sum(ones)/float(len(ones))
        accs.append(acc_)
        f1s.append(f1)
        perc_ones.append(perc_one)

    final_acc = 100*sum(accs)/len(accs)
    final_f1 = 100*sum(f1s)/len(f1s)
    final_ones = 100*sum(perc_ones)/len(perc_ones)

    binary_label_count = convert_label_count_to_binary_label(label_count, label_space)
    true_positive_rate = 100*binary_label_count[1] / float(binary_label_count[1] + binary_label_count[0])
    final_er = abs(true_positive_rate - final_ones)

    return final_acc, final_f1, final_ones, final_er

@torch.no_grad()
def pred_summac(model, documents, answer_sent, args):
    scores = model.score([''.join(documents)], [answer_sent])['scores']
    entailment_idx = model_map[args.model.split('_')[-1]]['entailment_idx']
    score = scores[entailment_idx]

    if args.use_threshold:
        output = score > args.threshold
        one = score > args.threshold
        
    else:
        raise NotImplementedError
    
    return output, one


@torch.no_grad()
def pred_encoder_models(sequences, tokenizer, model, device, args):
    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    for k, v in batch.items():
        batch[k] = v.to(device)
    probs = torch.softmax(model(**batch).logits, dim=1)
    
    if args.model == "cross-encoder/nli-deberta-v3-base":
        # pred = (1 in out) # whether entail
        if args.use_threshold:
            pred = torch.count_nonzero((probs[:, 1] > args.threshold).long()).item() > 0
        else:
            pred = (1 in out)
    elif args.model == 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli':
        # pred = (0 in out)
        if args.use_threshold:
            pred = torch.count_nonzero((probs[:, 0] > args.threshold).long()).item() > 0
        else:
            pred = (0 in out)
    elif args.model == 'google/t5_xxl_true_nli_mixture':
        if args.use_threshold:
            pred = torch.count_nonzero((probs[:, 1] > args.threshold).long()).item() > 0
        else:
            pred = (1 in out)

    output = int(pred)
    one = int(pred) == 1
    return output, one

@torch.no_grad()
def pred_decoder_models(sequences, tokenizer, model, device, args):
    model.eval()
    pred = False
    input_texts = []
    for s in sequences:
        premise, hypothesis = s[0], s[1]
        input_text = "premise: %s hypothesis: %s"%(premise, hypothesis)
        input_texts.append(input_text)
    
        input_ids = tokenizer(
            input_text, padding=True, truncation=True, return_tensors="pt", max_length=128
        ).input_ids.to(device)
        decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids.to(device)[:, :1]
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        all_logits = torch.cat((outputs['logits'][:, 0, 3:4], outputs['logits'][:, 0, 209:210]), dim=-1)
        probs = torch.softmax(all_logits, dim=-1)
        
        pred_instance = torch.count_nonzero((probs[:, 1] > args.threshold).long()).item() > 0
        if pred_instance:
            pred = True
            break

        input_ids = input_ids.detach().cpu()
        decoder_input_ids = decoder_input_ids.detach().cpu()
        probs = probs.detach().cpu()


    output = int(pred)
    one = int(pred) == 1
    return output, one
    
def convert_label_count_to_binary_label(label_count, label_space):
    binary_label_count = {0: 0, 1: 0}
    for k, v in label_count.items():
        binary_label_count[label_space[k]] += v

    return binary_label_count
    

def load_model(args, device):
    if args.model == 'random':
        pass
    elif args.model == 'docnli':
        label_list = ["entailment", "not_entailment"]#, "contradiction"]
        num_labels = len(label_list)
        model = RobertaForSequenceClassification(num_labels)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
        load_ckpt(model, device, args)
        model.to(device)
    elif 'summac' in args.model:
        if 'summac-zs' in args.model:
            model_name = args.model.split('_')[-1]
            model = SummaCZS(granularity="sentence", model_name=model_name, device=device) # If you have a GPU: switch to: device="cuda"
        elif 'summac-conv' in args.model:
            model_name = args.model.split('_')[-1]
            model = SummaCConv(models=[model_name], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
            tokenizer = None
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.model == 'google/t5_xxl_true_nli_mixture':
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model)
        print('finish loading model')
        model.to(device)

    return model, tokenizer



@torch.no_grad()
def eval_doc_nli(model, tokenizer, device, label_space, f_in, data, args):
    label_list = ["entailment", "not_entailment"]
    num_labels = len(label_list)
    test_examples = load_NLIdataset(f_in['input_file'], label_space['partially'] == 1)
    print('num_labels:', num_labels,  ' test size:', len(test_examples))


    '''load test set'''
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, "classification",
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    '''
    start evaluate on dev set after this epoch
    '''
    model.eval()
    acc, f1, er, predicted_supported, preds = evaluation(test_dataloader, device, model, 1-f_in['positive_label'])  # should flip the positive label

    pred_dict = {}
    i = 0
    for key, inst in data.items():
        pred_dict[key] = []
        for pair in inst['hyp_label_pairs']:
            pred_dict[key].append({"answer": pair['answer'], "pred": int(preds[i])})
            i += 1

    return acc, f1, er, predicted_supported, pred_dict

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)

    if args.model != 'random':
        model, tokenizer = load_model(args, device)
        model.eval()

    label_spaces = [{'partially':0, 'supported':1, 'not_supported':0}]
    print('input_file', args.input_file)
    
    f1_across_files = []
    er_across_files = []
    acc_across_files = []
    predicted_supported_across_files = []
    predictions_across_files = {}
    
    all_test_files = json.load(open(args.input_file))
    for f_in in all_test_files:
        data = json.load(open(f_in['input_file']))
        keys = data.keys()

        print('loading data....', f_in['input_file'])
        accuracy_list = []
        f1_list = []
        er_list = []
        predicted_supported_list = []

        for label_space in label_spaces:
            label_count = {'partially':0, 'supported':0, 'not_supported':0}

            if args.model == 'random':
                acc_, f1_, predicted_supported_, er_ = random_baseline(data, label_count, label_space, keys, f_in['positive_label'])
            elif args.model == 'docnli':
                acc_, f1_, er_, predicted_supported_, predictions = eval_doc_nli(model, tokenizer, device, label_space, f_in, data, args)
            else:
                outputs = []
                labels = []
                ones = []
                predictions = {}

                for key in tqdm(keys):
                    documents = data[key]['documents']
                    hyp_label_pairs = data[key]['hyp_label_pairs']
                    predictions[key] = []
                    for pair in hyp_label_pairs:
                        if args.calculate_score:
                            labels.append(label_space[pair['label']])
                            label_count[pair['label']] += 1                       

                        answer_sent = pair['answer']
                        if 'summac' in args.model:
                            output, one = pred_summac(model, documents, answer_sent, args)
                        else:
                            sequences = []
                            for s in documents:
                                if len(s.strip('<b>').strip('</b>').strip('Paragraph ')) <= 2:
                                    continue
                                sequences.append([s, answer_sent])
                            if 't5' in args.model:
                                output, one = pred_decoder_models(sequences, tokenizer, model, device, args)
                            else:
                                output, one = pred_encoder_models(sequences, tokenizer, model, device, args)

                        outputs.append(output)
                        ones.append(one)
                        predictions[key].append({"answer": answer_sent, "pred": output})

                        
                predicted_supported_ = 100*sum(ones)/float(len(ones))
                print('percentage predicted supported: %2.2f'%(predicted_supported_))

                if args.calculate_score:
                    acc_ = 100*acc(outputs, labels)
                    binary_label_count = convert_label_count_to_binary_label(label_count, label_space)
                    true_positive_rate = 100*binary_label_count[1] / float(binary_label_count[1] + binary_label_count[0])
                    er_ = abs(true_positive_rate - predicted_supported_)

                    f1_ = 100*f1_score(labels, outputs, pos_label= f_in['positive_label'], average='binary')
                    print(acc_, f1_, er_)
                    print('\n')

            if args.calculate_score:
                accuracy_list.append(acc_)
                f1_list.append(f1_)
                er_list.append(er_)
                predicted_supported_list.append(predicted_supported_)
        if args.model != 'random':
            predictions_across_files.update(predictions)
        if args.calculate_score:
            if len(accuracy_list) == 2:
                print('[Acc] partial as not_supported: %2.2f , partial as supported: %2.2f | \
                        [F1] partial as not_supported: %2.2f , partial as supported: %2.2f | \
                    [ER] partial as not_supported: %2.2f , partial as supported: %2.2f ' \
                    %(accuracy_list[0], accuracy_list[1], f1_list[0], f1_list[1], er_list[0], er_list[1]))
            else:
                print('[Acc]: %2.2f | [F1]: %2.2f | [ER]: %2.2f | [Pred Supported]: %2.2f'%(accuracy_list[0], f1_list[0], er_list[0], predicted_supported_list[0]))

            f1_across_files.append(f1_list[0])
            er_across_files.append(er_list[0])
            acc_across_files.append(accuracy_list[0])
            predicted_supported_across_files.append(predicted_supported_list[0])
        

    # after testing on all data
    if args.write_results: 
        with open(args.result_file, 'w') as f:
            f.write('%s'%args.model)
            f.write('\nF1\t')
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

        if args.prediction_file != None:
            with open(args.prediction_file, 'w') as f:
                f.write(json.dumps(predictions_across_files, indent=4))

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)

    parser.add_argument("--input_file", default="data.json")
    parser.add_argument("--initialize_model_from_checkpoint", type=str, default=None)
    parser.add_argument("--calculate_score", action='store_true')
    parser.add_argument("--use_threshold", action='store_true')
    parser.add_argument("--write_results", action='store_true')
    parser.add_argument("--result_file", type=str, default="results.csv")
    parser.add_argument("--prediction_file", type=str, default="predictions.csv")

    # doc NLI
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args()

    main(args)