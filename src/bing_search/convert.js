const axios = require('axios');
const { Readability } = require('@mozilla/readability');
const JSDOM = require('jsdom').JSDOM;
const fs = require('fs');


const mode = '10_q'
const inputFile = 'retrieval_results/urls/urls_' + mode + '.json';

const rootDir = './raw_html_pages/' + mode;

if (!fs.existsSync(rootDir)) {
  fs.mkdirSync(rootDir);
}

console.log(`Starting...`)
fs.readFile(inputFile, 'utf-8', (error, data) => {
  if (error) {
    console.error(`Error reading file ${inputFile}: ${error}`);
    return;
  }

  const urlLists = JSON.parse(data);
  

  urlLists.forEach((urls, q_index) => {
    console.log(`URL list ${q_index + 1}:`);
    console.log(urls);
    var html_list = [];
    const outputDir = `${rootDir}/q_${q_index}`
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir);
    }

    urls.forEach((url, index) => {
      const outputFile = `${outputDir}/output${index}.txt`;

      axios.get(url)
        .then(response => {
          const dom = new JSDOM(response.data, { url });
          const article = new Readability(dom.window.document).parse();
          fs.writeFileSync(outputFile, article.content);
          console.log(`Successfully wrote text content for ${url} to ${outputFile}`);

        })
        .catch(error => {
          console.error(`Error fetching HTML content for ${url}: ${error}`);
        });
    });


  });

    
});


  