#!/bin/bash

# Arguments 
# input_file      - the json file containing paths and positive label of different settings
#                   An example is shown in "input_files.json"
# threshold       - the thresold chosen for binary judgement
# result_file     - path to the results (reporting F1/Acc/ER)
# prediction_file - path to prediction file (just saving the model predictions)
# calculate_score - whether to compute the scores instead of just saving the predictions
# write_results   - whether to save the results 
# use_threshold   - whether to use a threshold, or just taking the max of the logits


# Should specify which model to use
# Model should be one of the following
# model="cross-encoder/nli-deberta-v3-base"
# model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# model="google/t5_xxl_true_nli_mixture"
# model="docnli"
# model="summac-conv_vitc"
# model="random"
echo $model

threshold=0.2
python auto_supported.py \
        --model ${model} \
        --input_file input_file.json \
        --threshold $threshold \
        --result_file results/results.csv \
        --prediction_file predictions/pred_${threshold}.json \
        --calculate_score \
        --use_threshold \
        --write_results \
        --initialize_model_from_checkpoint /path/to/docnli/checkpoint
