import os
import time
import signal
import requests
from htmldate import find_date

excluded_sites = [
                  "youtube.com",
                  ".pdf",
                  "quora.com",
                  "reddit.com"
                  ]



PLACE_HOLDER = {
    "entities_info": [
        {
            "name": "placeholder",
            "description": "placeholder"
        }
    ],
    "pages_info": [
        {
            'page_name': 'placeholder',
            'page_url': 'placeholder',
            'page_timestamp': 'placeholder',
            'page_snippet': 'placeholder',
        }
    ]
}


def timeout_handler(num, stack):
    print("received sigalrm")
    raise Exception("Time out!")


class WebRetriever:

    def __init__(self, engine: str, answer_count: int = 10):
        self.engine = engine
        if engine == 'bing':
            # self.subscription_key = os.environ[
                # 'BING_SEARCH_V7_SUBSCRIPTION_KEY']
            self.subscription_key = "bd965b8a138f47ec9cbf3a906892ccc1"
            self.endpoint = os.environ['BING_SEARCH_V7_ENDPOINT']
            self.mkt = 'en-US'
            self.answer_count = answer_count
            self.headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}
        elif engine == 'google':
            pass

    def get_results(self, query: str, time_stamp=None, raw_count=20, offset=0):
        if self.engine == 'bing':
            params = {
                'q': query,
                'mkt': self.mkt,
                'count': raw_count,
                'offset': offset,
                'freshness': "2000-01-01..{}".format(time_stamp) if
                time_stamp else None
            }
            print(params)
            try:
                start = time.time()
                response = requests.get(self.endpoint,
                                        headers=self.headers,
                                        params=params
                                        )
                response.raise_for_status()
                end = time.time()
                print('search finished, time take:', end - start)
            except Exception as e:
                print('exception happens during the search:')
                print(e)
                return PLACE_HOLDER
            response_json = response.json()
            entities_info = []
            pages_info = []

            if 'entities' in response_json.keys():
                entities_info = []
                for info in response_json['entities']['value']:
                    entities_info.append(
                        {
                            'name': info['name'],
                            'description': info['description']
                        }
                    )
            if 'webPages' not in response_json:
                return PLACE_HOLDER
            retrieved_pages = response_json['webPages']['value']
            count = 0
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            for i, page in enumerate(retrieved_pages):
                page_info_entry = {
                    'page_name': None,
                    'page_url': None,
                    'page_timestamp': None,
                    'page_snippet': None
                }
                if not self.url_is_excluded(page['url']):
                    page_info_entry['page_name'] = page['name']
                    page_info_entry['page_url'] = page['url']
                    page_info_entry['page_snippet'] = page['snippet']

                    try:
                        page_date = find_date(page['url'], verbose=False, original_date=False)
                        page_info_entry['page_timestamp'] = page_date
                    except Exception as ex:
                        print('Error happens during extracting page timestamp:')
                        print(ex)
                    pages_info.append(page_info_entry)
                    count += 1
                    if count == self.answer_count:
                        break
            return {
                "entities_info": entities_info,
                "pages_info": pages_info
            }

    @staticmethod
    def url_is_excluded(url):
        for ex_url in excluded_sites:
            if ex_url in url:
                return True
        return False

