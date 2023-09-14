import html2text
import os
import json
from tqdm import tqdm
import re

def write_json(filename, data):
    fw = open(filename, 'w')
    fw.write(json.dumps(data, indent=4))
    fw.close()

NUM_EX=271

for ANSWER_COUNT in [10]:
    for answer_source in ['q']:
        data = json.load(open('retrieval_results/raw_results/bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source)))

        print('finish loading data...', len(data))
        rootdir = 'raw_html_pages/%d_%s'%(ANSWER_COUNT, answer_source)
        folders = os.listdir(rootdir)
        assert len(folders) == NUM_EX, len(folders)
        
        for folder in tqdm(folders):
            example_index = int(folder.split('_')[1])
            files = os.listdir(os.path.join(rootdir, folder))
            for f in files:
                page_index = int(f.split('.')[0].split('output')[1])
                string = open(os.path.join(rootdir, folder, f)).read()

                h = html2text.HTML2Text()
                h.ignore_links = True
                parsed_string = h.handle(string).replace(u"\u2018", "'").replace(u"\u2019", "'")

                num_subs = 0
                
                parsed_string, num_subs = re.subn(r'!\s?\[[^\(\)\[\]]*\]\s?\([^\(\)\[\]]*\)', '', parsed_string, flags=re.DOTALL)
                parsed_string, num_subs = re.subn(r'http(s?)://[^\s\n]*(\s|\n)', ' ', parsed_string, flags=re.DOTALL)

                data[example_index]['pages'][page_index]['page_text'] = parsed_string

            # filter
            pages_with_text = []
            for p in data[example_index]['pages']:
                if 'page_text' in p:
                    pages_with_text.append(p)

                if len(pages_with_text) >= ANSWER_COUNT:
                    break
            data[example_index]['pages'] = pages_with_text

        write_json('retrieval_results/parsed_results/parsed_bing_results_%d_%s.json'%(ANSWER_COUNT, answer_source), data)
