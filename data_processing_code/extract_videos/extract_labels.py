import pandas as pd
import numpy as np
import os, shutil, glob
import re
import pdb

#import cv2 as cv
#import skvideo.io
#from moviepy.editor import *

import random

import warnings
warnings.filterwarnings("ignore")

# load required local modules
from util import *
# specify the paths to the data
from variables import *

save_dic = {}
def main(video_name):
	print 'Processing video: {}'.format(video_name)

	# load data
	actions_data, bbs_data = read_csv_data(data_dir+action_path, data_dir+bb_path)


	game_bbs = bbs_data[bbs_data.youtube_id == video_name]
	game_actions = actions_data[actions_data.youtube_id == video_name]
	game_actions = game_actions.reset_index()
	game_actions.event_end *= 0.001
		
	save_dic[video_name] = {}
	
	#import cPickle as Pickle
	#Pickle.dump(actions_data['event'].unique(), open('/ais/gobi4/basketball/olga_ethan_features/events.pkl', 'w'))
	#import pdb
	#pdb.set_trace()

	# iterating over clips/events
	for clip_ind, clip in game_actions.iterrows():

		#clip_dir = setFileDirectory(video_directory, 'clip_{}'.format(clip_ind+1))
		save_dic[video_name]['clip_{}'.format(clip_ind+1)] = clip['event']
			

if __name__ == '__main__':
	
	actions_data, bbs_data = read_csv_data(data_dir+action_path, data_dir+bb_path)
	videos_id = actions_data.youtube_id.unique()
	for vid in videos_id:
		try:
			main(vid)
		except Exception as e:
			print 'Error in processing: {}'.format(vid)
			print e
			
	import cPickle as Pickle
	Pickle.dump(save_dic, open('/ais/gobi4/basketball/olga_ethan_features/event_labels.pkl', 'w'))