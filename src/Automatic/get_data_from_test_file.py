import json
from collections import Counter
from typing import Any, Dict, List, Optional

from nltk.tokenize import sent_tokenize


def _get_sentence_tokenized_docs(
        docs_data: List[Dict[str, Any]],
        question_id: int) -> List[str]:

    docs_entries = list(filter(lambda entry: entry['question_id'] == question_id, docs_data))
    if len(docs_entries) != 1:
        raise ValueError(f'Expected 1 entry with given question id, found {len(docs_entries)}.')
    docs_entry = docs_entries[0]

    sentences: List[str] = []

    # docs = docs_entry['docs']
    # title_to_text_count_map = Counter()
    # for doc in docs:
    #     title = doc['title']
    #     title_to_text_count_map[title] += 1

    #     text = doc['text'].strip('Quote: ')
    #     paragraph_num = title_to_text_count_map[title]

    #     if paragraph_num == 1:
    #         doc_num = len(title_to_text_count_map)
    #         title_sentence = f'<b>[Document {doc_num}]: {title}</b>'
    #         sentences.append(title_sentence)

    #     paragraph_sentence = f'<b>Paragraph {paragraph_num}</b>'
    #     sentences.append(paragraph_sentence)

    #     sentences += sent_tokenize(text)

    prev_title: Optional[str] = None
    doc_num: int = 0
    paragraph_num: int = 0
    for doc in docs_entry['docs']:
        title = doc['title']
        if title != prev_title:
            prev_title = title
            doc_num += 1
            paragraph_num = 0
            title_sentence = f'<b>[Document {doc_num}]: {title}</b>'
            sentences.append(title_sentence)

        paragraph_num += 1
        paragraph_sentence = f'<b>Paragraph {paragraph_num}</b>'
        sentences.append(paragraph_sentence)

        text = doc['text'].strip('Quote: ')
        sentences += sent_tokenize(text)

    return sentences


def _get_hypothesis_label_pairs(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hypothesis_label_pairs: List[Dict[str, Any]] = []
    for i_annotation, annotation in enumerate(annotations):
        labels: List[str] = annotation['labels']
        most_common_label_and_count = Counter(labels).most_common(1)[0]
        if most_common_label_and_count[1] == 1:
            # All labels are different, discard this example
            continue

        sentence_num = i_annotation
        hypothesis_label_pairs.append({
            'label': most_common_label_and_count[0],
            'answer': annotation['answer'],
            'annotations': annotation['labels'],
            'answer_id': str(sentence_num)
        })

    return hypothesis_label_pairs


def _generate_nli_input(
        docs_data: List[Dict[str, Any]],
        annotations_data: List[Dict[str, Any]],
        prediction_key_template: str) -> Dict[str, Any]:

    nli_input: Dict[str, Any] = {}

    for annotations_entry in annotations_data:
        question_id = annotations_entry['question_id']

        document_sentences = _get_sentence_tokenized_docs(docs_data, question_id)
        hypothesis_label_pairs = _get_hypothesis_label_pairs(annotations_entry['annotations'])

        key = prediction_key_template.replace('{QUESTION_ID}', str(question_id))
        nli_input[key] = {
            'documents': document_sentences,
            'hyp_label_pairs': hypothesis_label_pairs
        }

    return nli_input


def get_data_from_test_file(input_info):
    if 'input_file' in input_info:
        print('loading data....', input_info['input_file'])

        # Legacy format
        with open(input_info['input_file'], 'r', encoding='UTF-8') as input_file:
            return json.load(input_file)

    print('loading data....', input_info['annotations_path'], input_info['docs_path'])

    annotations_data = []
    with open(input_info['annotations_path'], 'r', encoding='UTF-8') as input_file:
        annotations_data = json.load(input_file)

    docs_data = []
    with open(input_info['docs_path'], 'r', encoding='UTF-8') as input_file:
        docs_data = json.load(input_file)

    prediction_key_template: str = input_info['prediction_key_template']

    return _generate_nli_input(docs_data, annotations_data, prediction_key_template)
