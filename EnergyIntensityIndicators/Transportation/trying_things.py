from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import requests
import pandas as pd

tedb_url = 'https://tedb.ornl.gov/wp-content/uploads/2020/06/TEDB_38.1_Spreadsheets_06242020.zip'
# r = requests.get(tedb_url)
# with zipfile.ZipFile.open(r, mode=r) as zip_ref:
#     print(zip_ref)
date = '04302020'
table_number = '1_03'
file_url = f'https://tedb.ornl.gov/wp-content/uploads/2020/04/Table{table_number}_{date}.xlsx'

# resp = urlopen(tedb_url)
# zipfile = ZipFile(BytesIO(resp.read()))
# for line in zipfile.open(zipfile).readlines:
#     print(line.decode('utf-8'))


# r = requests.get(file_url)
# this = r.content
# print(this, type(r))

xls = pd.read_excel(file_url, sheetname=None, header=11) 
print(xls)