from nltk.tokenize import sent_tokenize

import logging
import argparse
import json
import os
import torch
from model_summac import SummaCZS, SummaCConv
from tqdm import tqdm
import random
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
    # out = torch.softmax(model(**batch).logits, dim=1).max(dim=1)[1]
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
    batch_size = 3
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
    elif 'gpt3' in args.model:
        pass
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.model == 'google/t5_xxl_true_nli_mixture':
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model.to(device)

    return model, tokenizer


@torch.no_grad()
def main(args):
    # load WebGPT data and compute NLI score based on some answer and some documents 
    data = json.load(open(args.input_file))['data']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)

    print('loading model', 'doc_type:', args.doc_type)
    model, tokenizer = load_model(args, device)

    if args.doc_type == 'webgpt':
        # answer_keys = ['003_answer', 'gpt3_wdoc_answer', 'gpt3_randdocs_answer', 'gpt3_bingdoc_detokenized_answer', 'alpaca_answer', 'alpaca_wdoc_answer', 'alpaca_randdocs_answer', 'alpaca_bingdoc_detokenized_answer']
        answer_keys = ['gpt3_humandoc_answer', 'alpaca_humandoc_answer']
    elif args.doc_type == 'random':
        answer_keys = ['003_answer', 'gpt3_wdoc_answer', 'gpt3_randdocs_answer', 'gpt3_bingdoc_detokenized_answer', 'alpaca_answer', 'alpaca_wdoc_answer', 'alpaca_randdocs_answer', 'alpaca_bingdoc_detokenized_answer']
        # answer_keys = ['gpt3_humandoc_answer', 'alpaca_humandoc_answer']
    elif args.doc_type == 'bing':
        # answer_keys = ['003_answer', 'gpt3_wdoc_answer', 'gpt3_randdocs_answer', 'gpt3_bingdoc_detokenized_answer', 'alpaca_answer', 'alpaca_wdoc_answer', 'alpaca_randdocs_answer', 'alpaca_bingdoc_detokenized_answer']
        answer_keys = ['gpt3_humandoc_answer', 'alpaca_humandoc_answer']
    elif args.doc_type == 'human':
        # answer_keys = ['003_answer', 'gpt3_wdoc_answer', 'gpt3_randdocs_answer', 'gpt3_bingdoc_detokenized_answer', 'gpt3_whudoc_answer', 'alpaca_answer', 'alpaca_wdoc_answer', 'alpaca_randdocs_answer', 'alpaca_bingdoc_detokenized_answer', 'alpaca_whudoc_answer']
        answer_keys = ['gpt3_humandoc_answer', 'alpaca_humandoc_answer']
    else:
        raise NotImplementedError
    outputs = {key: [] for key in answer_keys}

    print('doing doc type: %s'%(args.doc_type))
    print(answer_keys)

    for inst in tqdm(data):
        webgpt_docs = inst['positive_ctxs']
        rand_docs = inst['random_ctxs']
        bing_docs = inst['cleaned_bing_ctxs']
        human_docs = inst['human_ctxs']

        if args.doc_type == 'webgpt':
            docs = webgpt_docs
        elif args.doc_type == 'random':
            docs = rand_docs
        elif args.doc_type == 'bing':
            docs = bing_docs
        elif args.doc_type == 'human':
            docs = human_docs
        else:
            raise NotImplementedError
        
        doc_sentences = []
        for d in docs:
            doc_sentences += (sent_tokenize(d['text'].strip('Quote: ')))
        # for d in rand_docs:
        #     rand_doc_sentences += (sent_tokenize(d['text'].strip('Quote: ')))

        # gpt3_answer = inst['003_answer']
        # gpt3_wdoc_answer = inst['gpt3_wdoc_answer']
        # gpt3_randdocs_answer = inst['gpt3_randdocs_answer']
        # gpt3_bingdoc_answer = inst['gpt3_bingdoc_answer']

        # alpaca_answer = inst['alpaca_answer']
        # alpaca_wdoc_answer = inst['alpaca_wdoc_answer']
        # alpaca_randdocs_answer = inst['alpaca_randdocs_answer']
        # alpaca_bingdoc_answer = inst['alpaca_bingdoc_answer']

        for key in answer_keys:
            for answer_sent in sent_tokenize(inst[key]):
                sequences = []
                for s in doc_sentences:  # each (documents, sentence) pair
                    sequences.append([s, answer_sent])
                
                if 't5' in args.model:
                    output, one = pred_decoder_models(sequences, tokenizer, model, device, args)
                else:
                    output, one = pred_encoder_models(sequences, tokenizer, model, device, args)
                
                outputs[key].append(output)

    for key in answer_keys:
        print('predicted positive percentage for key [%s]'%(key), '%2.2f'%(sum(outputs[key])/float(len(outputs[key]))))

                

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5_xxl_true_nli_mixture", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)

    parser.add_argument("--input_file", default="data.json")
    parser.add_argument("--initialize_model_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_threshold", action='store_true')
    parser.add_argument("--doc_type", default='webgpt', type=str)
    args = parser.parse_args()

    main(args)