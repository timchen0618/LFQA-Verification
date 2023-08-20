from factscore.factscorer import FactScorer
from tqdm import tqdm
import openai
import math

## The knolwedge source should be ready in a .jsonl format, where each line is a dictionary containing title and text. 
## text can either be a string or a list of strings (e.g., sections).
import json
import os

answer_source = 'alpaca_wdoc'
nli_data = json.load(open('../../Data/nli_data/main-nli-all100_%s.json'%(answer_source)))
path_to_jsonl_file = 'knowledge/nli/jsonl/nli_knowledge_webgpt.jsonl'
name_of_your_knowledge_source = 'nli_knowledge_webgpt' 
# path_to_jsonl_file = 'knowledge/nli/jsonl/nli_knowledge_human.jsonl'
# name_of_your_knowledge_source = 'nli_knowledge_human' 


# command = 'jsonl'
command = 'score'



if command == 'jsonl':
    # Create Jsonl File
    
    title2docs = {}
    for k in tqdm(nli_data.keys()):
        id_ = int(k.split('-')[-1])
        # path_to_jsonl_file = 'knowledge/nli/jsonl/nli_knowledge_webgpt_%d.jsonl' % id_
        doc_data = []
        paragraph_text = ""
        title_ = "%s_%d" % (name_of_your_knowledge_source, id_)
        title2docs[title_] = []
        for i, d in enumerate(nli_data[k]['documents']):

            if d[:12] == '<b>[Document': # doc title
                titleid = int(d.strip('<b>[Document').split(']')[0].strip())
                if titleid != 1:
                    title2docs[title_].append(paragraph_text)

            elif d[:12] == '<b>Paragraph':
                p_id = int(d.strip('<b>Paragraph').strip('</b>').strip())
                if p_id != 1:
                    title2docs[title_].append(paragraph_text)
                paragraph_text = ""

            else:
                paragraph_text += d
        title2docs[title_].append(paragraph_text)

    with open(path_to_jsonl_file, 'w') as f:
        # for l in all_doc_data:
        for title, docs in title2docs.items():
            l = {"title": title, "text": docs}
            f.write(json.dumps(l))
            f.write('\n')    

elif command == 'score':
    openai.organization = "org-Bs2FYbWKaWDsgSyc2j9Uy3Gp"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    fs = FactScorer(data_dir='/data/hungting/projects/LFQA/FActScore/', model_name='retrieval+llama+npm', openai_key='.openaikey')

    path_to_output_db_file = 'knowledge/nli/db/%s.db' % (name_of_your_knowledge_source)
    # # this will create a database using your file
    # # for English Wikipedia (18GB)), it takes ~8 hours
    # # once DB file is created, you can reuse it by only specifying `db_path`
    fs.register_knowledge_source(name_of_your_knowledge_source,
                                    data_path=path_to_jsonl_file,
                                    db_path=path_to_output_db_file)
    
    scores = []
    score_to_save = {}
    num_facts_per_responses = []
    for k in tqdm(nli_data.keys()):
        id_ = int(k.split('-')[-1])
        # id_ = int(k)
        topic = "%s_%d" % (name_of_your_knowledge_source, id_)
        score_to_save[k] = []
        # name_of_your_knowledge_source = 'nli_knowledge_webgpt_%d' % id_
        # path_to_jsonl_file = 'knowledge/nli/jsonl/nli_knowledge_webgpt_%d.jsonl' % id_
        # now, when you compute a score, specify knowledge source to use
        generations = []
        topics = []
    # for k in tqdm(nli_data.keys()):
        # id_ = int(k.split('-')[1])
        # name_of_your_knowledge_source = 'nli_knowledge_webgpt_%d' % id_
        for pairs in nli_data[k]['hyp_label_pairs']:
            answer = pairs['answer']
            generations.append(answer)
            topics.append(topic)
        
            out = fs.get_score([topic], [answer], knowledge_source=name_of_your_knowledge_source)
            if math.isnan(out['score']):
                out['score'] = 0
                out['num_facts_per_response'] = 0
            
            scores.append(out["score"])
            score_to_save[k].append(out["score"])
            num_facts_per_responses.append(out["num_facts_per_response"])


    print(scores)
    print('FActScore: %.4f' % (sum(scores)/len(scores)))
    print('Avg. # of facts per response: %.4f' % (sum(num_facts_per_responses)/len(num_facts_per_responses)))
    score_to_save['avg'] = sum(scores)/len(scores)

    fw = open('scores/%s.json'%(answer_source), 'w')
    fw.write(json.dumps(score_to_save, indent=4))
    fw.close()