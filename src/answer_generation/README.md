# Retrieval-Augmented Generation of Answers

This directory contains code to generate answers given questions, evidence
documents and the template for a prompt.

## Data

Questions, with their corresponding documents, should be in the following format:
```
[
    {
        "question_id": <unique integer id>,
        "question": "<question>",
        "docs": [
            {
                "doc_id": "<unique string id>",
                "title": "<title>",
                "text": "<document text>"
            },
            ...
        ]
    },
    ...
]
```
The files used in our experiments can be found at `~/data/docs`.

Prompt templates can be arbitrary text. They support the `{DOCUMENTS}`
and `{QUESTION}` placeholders for documents and questions respectively.
The files used in our experiments can be found at `~/data/prompts`.

## Generating Answers

Answers can be generated using with `prompt_lm.py`.

The
[Alpaca](https://github.com/tatsu-lab/stanford_alpaca/) model can be run once
weights are stored locally as follows:
```
python prompt_lm.py \
        -i input/docs/file.json \
        -p prompt/template/file.json \
        -o output/file.json \
        -m path/to/alpaca/weights/ \
        --prompt_docs_delim '\n'
``` 
The `prompt_docs_delim` parameter configures the delimiter used between documents
when they are added to the prompt.

OpenAI models can be run as follows:
```
python prompt_lm.py \
        -i input/docs/file.json \
        -p prompt/template/file.json \
        -o output/file.json \
        -m openai_model_name \
        --prompt_docs_delim '\n' \
        --openai_model \
        --org org_yourorg
``` 