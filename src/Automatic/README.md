# Automatically Identifying Unsupported Parts

## Data 
You should store the labels, answer sentences and documents in the following format. 
```
{
    "key_0": {
        "documents": [
        ],
        "hyp_label_pairs": [
            {
                "label": "supported",
                "answer": "",
                "answer_id": "0"
            },
            {
                "label": "not_supported",
                "answer": "",
                "answer_id": "1"
            },
            ... 
        ]
    },
    "key_1": {
        ...
    }
    ...
}
```
And you should put the file paths to all the data you would like to evaluate on in one json file, formatted like this: (See `input_file.json` for an example.)
```
[  
    {
        "input_file" :"/path/to/data/",
        "positive_label": 0 or 1
    },
    ...
]
```


## Running Inference 
Once you have the data ready and put the file paths (and the corresponding positive label) to `input_file.json`, you could run inference with `auto_supported.py`.  
An example script is as follows:
```
python nli.py \
        --model google/t5_xxl_true_nli_mixture \
        --input_file input_file.json \
        --threshold $threshold \
        --result_file results/results.csv \
        --prediction_file predictions/pred_${threshold}.json \
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
And the keys should match all the keys in the files of the `--input_file` argument you specify.

An example script is as follows:
```
python eval.py --pred_file predictions/pred.json \
               --output_file results/results.json \
               --input_file input_file.json
```

