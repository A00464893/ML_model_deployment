import matplotlib.pyplot as plt
import pandas as pd
import requests

url = 'https://data.nasdaq.com/api/v3/datasets/WIKI/FB.json'
nasdaq_api_key = "zsipeWww-UyVVqGysBYe"
r_data = requests.get(url, params=dict(api_key=nasdaq_api_key), ).json()
r_data_df = pd.DataFrame(r_data['dataset']['data'], columns=r_data['dataset']['column_names'])
r_data_df.sort_values(by="Date", inplace=True)
# print(r_data_df)
r_data_df.plot("Date", "Close", kind="line")
plt.show()
