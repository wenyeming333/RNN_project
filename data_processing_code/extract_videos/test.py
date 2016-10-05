import requests
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_color_codes()
sns.set_style("white")

url = "http://stats.nba.com/stats/locations_getmoments/?eventid=308&gameid;=0041400235"

# Get the webpage
response = requests.get(url)
# Take a look at the keys from the dict
# representing the JSON data
response.json().keys()
