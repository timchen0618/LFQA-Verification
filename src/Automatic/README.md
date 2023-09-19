# Automatically Identifying Unsupported Parts

## Data 
You should put the file paths to all the data you would like to evaluate on in one json file, formatted like this: (See `input_file.json` for an example).
```
[  
    {
        "annotations_path": "/path/to/annotations/file.json",
        "docs_path": "/path/to/docs/file.json",
        "prediction_key_template": "some-prefix-{QUESTION_ID}_some_suffix",
        "positive_label": 0 or 1
    },
    ...
]
```
The string `{QUESTION_ID}` is a placeholder in the `prediction_key_template` field that is replaced with the question id of the question being considered.

## Running Inference 
Once you have the data (annotations and docs files) and `input_file.json` ready, you could run inference with `auto_supported.py`.  
An example script is as follows:
```
python auto_supported.py \
        --model google/t5_xxl_true_nli_mixture \
        --input_file input_file.json \
        --threshold $threshold \
        --result_file results.csv \
        --prediction_file pred_${threshold}.json \
        --calculate_score \
        --use_threshold \
        --write_results 
```
You could also find an example and more details in `auto_supported.sh`.

## Evaluation
If you have just the prediction but did not compute the score earlier, you could evaluate the prediction with `eval.py`. The prediction should be in the format of: 
```
{
    "key_0": [
        {
            "pred": 0 or 1
        },
        ...
    ],
    "key_1": [], 
    ...
}
```
The keys should match the `prediction_key_template` templates in the entries of the `--input_file` argument you specify.

An example script is as follows:
```
python eval.py --pred_file predictions/pred.json \
               --output_file results/results.tsv \
               --input_file input_file.json
```

