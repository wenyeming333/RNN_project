
# coding: utf-8

# In[ ]:

from __future__ import unicode_literals
from __future__ import unicode_literals
import youtube_dl
import csv
import numpy as np
#from joblib import Parallel, delayed

# directory paths
filename = '/ais/gobi4/basketball/data/bball_dataset_april_4.csv'
output_dir = '/ais/gobi4/basketball/data/videos/'
num_cores = 10

# read the csv file

datta =[]
with open(filename, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        datta.append(row)    


# create the youtube urls

url_prefix = 'http://www.youtube.com/watch?v='
url_whole = []
output_names = []
height = []
width = []
for temp in datta:
    url_whole.append(url_prefix+temp[0])
    height.append(temp[2])
    width.append(temp[1])
    output_names.append(output_dir+temp[0]+'.%(ext)s')    

# turn them to numpy arrays
temp = np.unique(np.asarray(url_whole), return_index=True)
url_whole = temp[0].tolist()
height = np.array(height)[temp[1]].tolist()
width = np.array(width)[temp[1]].tolist()
output_names = np.array(output_names)[temp[1]].tolist()

failed = []
# download the videos
for ii in range(len(url_whole)):
    print 'Downloading Video ' + str(ii+1) + ' / ' + str(len(url_whole))
    ydl_opts = {
        'format': 'best[height='+height[ii]+'][width='+ width[ii]+']',
        'outtmpl': output_names[ii]
    }
    
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_whole[ii]])
    except:
        print 'error: ' + url_whole[ii] + ' failed!'
        failed.append(url_whole[ii])

        
print 'The following failed: '
for temp in failed:
    print temp
 


# In[6]:


# write the failed cases
with open(output_dir+'failed.txt', 'w') as ff:
    ff.write("\n".join(map(lambda x: str(x), failed)))

