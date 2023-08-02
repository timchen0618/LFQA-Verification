"""prompt open sourced model for answer generation"""
import json
from argparse import ArgumentParser
import os
from typing import Any, Dict, Iterable, List

import openai
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer)

QUESTION_PLACEHOLDER = '{QUESTION}'
DOCS_PLACEHOLDER = '{DOCUMENTS}'


def _generate_prompts(
        prompt_template: str,
        prompt_docs_delim: str,
        docs_data: List[Dict[str, Any]]) -> List[str]:

    prompts: List[str] = []
    for question_entry in docs_data:
        question = question_entry['question']
        docs = question_entry['docs']

        prompt_docs_substr = prompt_docs_delim.join([
            doc['text'] for doc in docs
        ])

        prompt = prompt_template.replace(
            QUESTION_PLACEHOLDER, question).replace(
            DOCS_PLACEHOLDER, prompt_docs_substr)

        prompts.append(prompt)

    return prompts


def _extract_answer(prompt: str, output: str) -> str:
    if prompt not in output:
        raise RuntimeError(f'Output does not start with prompt: {output} , {prompt}')

    answer_index = output.index(prompt) + len(prompt)
    return output[answer_index:]


def _prompt_local_lm(
        model_name: str,
        prompts: List[str],
        max_length: int,
        cuda_device: int,
        model_parallelism: bool) -> Iterable[str]:

    model_args = {}
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True

    device = 'cuda:{}'.format(cuda_device)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for prompt in tqdm(prompts, total=len(prompts)):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=max_length)

        output: str = tokenizer.batch_decode(gen_tokens)[0]
        yield _extract_answer(prompt, output)


def get_openai_response(
        prompt: str,
        model_name: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: int = 1,
        n: int = 1,
        logprobs: int = 1,
        stop: Any = None,
        echo: bool = True):
    response = openai.Completion.create(model=model_name,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    return response


def _prompt_openai_lm(
        model_name: str,
        prompts: List[str],
        max_tokens: int,
        organization: str) -> Iterable[str]:

    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = organization

    for prompt in tqdm(prompts, total=len(prompts)):
        response = get_openai_response(prompt, model_name, max_tokens)
        output = response['choices'][0]['text']
        yield _extract_answer(prompt, output)


def _prompt_lm(
        model_name: str,
        is_openai_model: bool,
        prompts: List[str],
        **kwargs) -> Iterable[str]:

    if is_openai_model:
        return _prompt_openai_lm(
            model_name,
            prompts,
            max_tokens=kwargs['max_tokens'],
            organization=kwargs['organization'])

    return _prompt_local_lm(
        model_name,
        prompts,
        max_length=kwargs['max_length'],
        cuda_device=kwargs['cuda_device'],
        model_parallelism=kwargs['model_parallelism'])


def prompt_lm(
        input_docs_path: str,
        prompt_file: str,
        prompt_docs_delim: str,
        output_answers_path: str,
        model_name: str,
        is_openai_model: bool,
        **kwargs):

    docs_data: List[Dict[str, Any]] = []
    with open(input_docs_path, 'r', encoding='UTF-8') as input_file:
        docs_data = json.load(input_file)

    prompt_template: str = ''
    with open(prompt_file, 'r', encoding='UTF-8') as input_file:
        prompt_template = ''.join(input_file.readlines())

    print('Generating prompts')
    prompts = _generate_prompts(prompt_template, prompt_docs_delim, docs_data)
    print(f'{len(prompts)} prompts generated')

    print(f"Processing {len(prompts)} prompts")

    answers_data: List[str] = []
    for question_entry, answer in zip(
            docs_data, _prompt_lm(model_name, is_openai_model, prompts, **kwargs)):

        answers_data_entry = {
            'question_id': question_entry['question_id'],
            'question': question_entry['question'],
            'answer': answer,
        }

        answers_data.append(answers_data_entry)
        with open(output_answers_path, 'w', encoding='UTF-8') as output_file:
            json.dump(answers_data, output_file, indent=4)


def main():
    argparse = ArgumentParser()
    argparse.add_argument('-i', "--input_docs_path", dest='input_docs_path', required=True)
    argparse.add_argument('-p', "--prompt_file", dest='prompt_file', required=True)
    argparse.add_argument('-o', "--output_answers_path", dest='output_answers_path', required=True)
    argparse.add_argument('-m', "--model", dest='model', required=True)
    argparse.add_argument('--openai_model', dest="is_openai_model", action='store_true')
    argparse.add_argument("--prompt_docs_delim", dest='prompt_docs_delim', default='\n')

    # Local LM args
    argparse.add_argument("--max_length", dest='max_length', default=1024, type=int)
    argparse.add_argument('-d', "--cuda_device", dest='cuda_device', default=0)
    argparse.add_argument("--model_parallelism", dest='model_parallelism', action='store_true')

    # OpenAI LM args
    argparse.add_argument("--max_tokens", dest='max_tokens', default=256, type=int)
    argparse.add_argument('--org', "--organization", dest='organization', default='')

    args = argparse.parse_args()
    # print('Args:', args)

    prompt_lm(args.input_docs_path,
              args.prompt_file,
              args.prompt_docs_delim,
              args.output_answers_path,
              args.model,
              args.is_openai_model,
              cuda_device=args.cuda_device,
              max_length=args.max_length,
              model_parallelism=args.model_parallelism,
              max_tokens=args.max_tokens,
              organization=args.organization)


if __name__ == "__main__":
    main()
