import os
import json
import requests

api_key = os.environ.get("EIA_API_Key")


api_category_ids = {'EIA Datsets': 371, 'seds_consumption_residential': 40209}
category = 'EIA Datsets'
call  = f'http://api.eia.gov/category/?api_key={api_key}&category_id={api_category_ids[category]}'

r = requests.get(call)
files = r.json()
print(files)