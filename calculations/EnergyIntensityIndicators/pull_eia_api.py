import os
import json
import requests

api_key = os.environ.get("api-token")

api_category_ids = {'seds_consumption_residential': 40209}
category = 'seds_consumption_residential'
call  = f'http://api.eia.gov/category/?api_key={api_key}&category_id={api_category_ids[category]}'

r = requests.get(call)
files = r.json()
print(files)