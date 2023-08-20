#!/bin/bash

# Arguments 
# input_file      - the json file containing all the question, context, and the corresponding generation, 
# threshold       - the thresold chosen for binary judgement
# use_threshold   - whether to use a threshold, or just taking the max of the logits


# Should specify which model to use
# Model should be one of the following
# model="cross-encoder/nli-deberta-v3-base"
# model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
model="google/t5_xxl_true_nli_mixture"
# model="docnli"
# model="summac-conv_vitc"
# model="random"

echo $model
for threshold in 0.3
do
    python supported_score.py \
                            --model ${model} \
                            --input_file all_data.json \
                            --use_threshold \
                            --threshold $threshold \
                            --initialize_model_from_checkpoint /data/hungting/models/DocNLI.pretrained.RoBERTA.model.pt \
                            --doc_type random

done
