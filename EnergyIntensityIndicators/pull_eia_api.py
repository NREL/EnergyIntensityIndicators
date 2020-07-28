import os
import json
import requests
import pandas as pd
import numpy as np

print('current directory', os.getcwd())
api_key = os.environ.get("EIA_API_Key")

api_category_ids = {'EIA Datsets': 371, 'seds_consumption_all': 40204, 'seds_consumption_residential': 40209}
category = 'seds_consumption_all'
call  = f'http://api.eia.gov/category/?api_key={api_key}&category_id={api_category_ids[category]}'

r = requests.get(call)
files = r.json()
print(files)

call2 = f'http://api.eia.gov/series/?api_key={api_key}&series_id=SEDS.PATCB.AL.A'
r = requests.get(call2)
j = r.json()
# df = pd.DataFrame.from_dict(j)
print(j)
# print(df)

consumption_all_btu = pd.read_csv('./EnergyIntensityIndicators/use_all_btu.csv')
print(consumption_all_btu.head())