# Readme for Bing Document Retrieval
## Setup 
You should instantiate a subscription for Bing Search API and obtain the subscription key. You should specify the subscription key as a environment variable `BING_SEARCH_V7_SUBSCRIPTION_KEY`, or put the following code block in your `~/.bashrc` file:
```
export BING_SEARCH_V7_SUBSCRIPTION_KEY="your_subscription_key"
export BING_SEARCH_V7_ENDPOINT="https://api.bing.microsoft.com/v7.0/search"
```
Now you are ready to retrieve webpages from Bing Search API. 

There are a few steps in retrieving relevant documents using Bing Search API: 
1. [Retrieving Webpages](#retrieving-webpages): Retrieve pages from Bing Search API using the question as the query. Save all the results, including the metadata in `retrieval_results/raw_results/`. 
2. [Getting the URLs](#getting-the-urls): Get the links from the previous steps and save them in a json file. 
3. [Create raw_html_pages and remove all webpages in raw_html_pages](#create-raw_html_pages-and-remove-all-webpages-in-raw_html_pages): Make sure the `raw_html_pages` folder is created and empty.
4. [Parse the Texts From the Pages](#parse-the-texts-from-the-pages): Post-process the raw html from retrieved pages using [readability.js](https://github.com/mozilla/readability) and [html2text](https://pypi.org/project/html2text/).
5. [Do the Second Stage of Retrieval](#do-the-second-stage-of-retrieval): Take the parsed documents and split them into a collection of 100-word segments. Retrieve top 4 documents using BM25 from all the segments. 


We will detail each step below:   

## Retrieving Webpages
Run in the current level of the repo
``` 
python retrive.py
```

The results are stored in `retrieval_results/raw_results/`.

## Getting the URLs
Run
```
python src_processing/get_urls.py 
```
The urls of the retrieved pages are stored in stored in retrieval_results/urls/.  

## Create raw_html_pages and remove all webpages in raw_html_pages 
```
mkdir -p raw_html_pages  
rm -r raw_html_pages/*
```

## Parse the Texts From the Pages
```
node convert.js
python src_processing/convert2text.py
```
The raw html pages scraped by `convert.js` will go to `raw_html_pages`.  
The parsed documents will be stored in `retrieval_results/parsed_results/`.  

## Do the Second Stage of Retrieval
```
python 2stage_retrieval.py
```

The retrieved documents will be stored in `retrieval_results/2stage_results/`.  
This will use the default hyperparameters. If you would like to use a different setting (e.g. different sized segments), please refer to the arguments in `2stage_retrieval.py`.  
The output file will be of the same format as in folder `data/docs/docs-bing.json`.  
