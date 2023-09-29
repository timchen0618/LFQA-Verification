# How Language Models Use In-context Evidence Documents to Generate Long-form Answers
This is the repository for the paper [How Language Models Use In-context Evidence Documents to Generate Long-form Answers]().

## Contents
1. [Requirements](#requirements)
2. [Collected Data](#collected-data)
3. [Reproduction](#reproduction)
4. [Citation](#citation)


## Requirements

Our code requires PyTorch (`torch`), HuggingFace Transformers (`transformers`) and the OpenAI API package (`openai`). Most of our experiments were run with `torch==2.0.1`, `transformers==4.30.1` and `openai==0.27.8` on Python 3.10.6.

## Collected Data

The [data folder](https://github.com/timchen0618/LFQA-Verification/tree/main/data)
contains:
- questions with corresponding human and model answers,
- evidence documents retrieved for each of the questions,
- prompt templates used for creating the prompts that were passed to the models, and
- human annotations of the attributability of each answer sentence to corresponding
evidence documents, for a subset of the question and models.

## Reproduction

### Prompting LMs

Details for how to reproduce our prompting of the LMs are in [`src/answer_generation`](https://github.com/timchen0618/LFQA-Verification/tree/main/src/answer_generation). Our setup is easily reusable with different questions, documents, prompts and/or models.

### Attribution Prediction
We benchmark several approaches on attribution of answer sentences using collected data. The details can be found in [`src/Automatic/`](https://github.com/timchen0618/LFQA-Verification/tree/main/src/Automatic).

### Retrieving Bing Documents

Steps for retrieving Bing evidence documents can be found in
[`src/bing_search`](https://github.com/timchen0618/LFQA-Verification/tree/main/src/bing_search).

## Citation