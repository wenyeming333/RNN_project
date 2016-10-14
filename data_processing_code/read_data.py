import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from variables import *

def split_train_val_test_index(total_number_videos,seed):
	total = np.arange(total_number_videos)
	train,temp = train_test_split(total,test_size=0.175,random_state=seed)
	val,test = train_test_split(temp,test_size=0.266,random_state=seed)
	return train,val,test

def get_clips(video_dir):
	res = []
	for directory in video_dir:
		res += [os.path.join(directory,files) for files in os.listdir(directory) if 'clip' in files]
	return sorted(res)

class DataSet(object):
	def __init__(self,actions,bbs):
		actions = actions.sort_values(['youtube_id', 'clip_start', 'event_end'])
		self.videos_list = actions.youtube_id.unique()
		train, val, test = split_train_val_test_index(257,42)
		train_videos, val_videos, test_videos = videos_list[train], videos_list[val], videos_list[test]
		train_dir = [os.path.join(processed_dir,video) for video in sorted(train_videos)]
		val_dir = [os.path.join(processed_dir,video) for video in val_videos.sort()]
		test_dir = [os.path.join(processed_dir,video) for video in test_videos.sort()]
		train_files = get_clips(train_dir)
		val_files = get_clips(val_dir)
		test_files = get_clips(test_dir)
		self.files = [train_files,val_files,test_files]
		self.number_files = [len(train_files),len(val_files),len(test_files)]
		self._epochs_completed = 0
		self._index_in_epoch = 0

	# If it is training, flag=1, 2 for validation, 3 for test.
	def next_batch_train(self,batch_size,flag=1):
		strat = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self.number_files[flag-1]:
			self._epochs_completed += 1
			# Shuffle the data
			if flag == 1:
				perm = np.arange(self.number_files[0])
				np.random.shuffle(perm)
				self.train_files = self.train_files[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self.number_files[0]
		end = self._index_in_epoch
		return self.files[flag-1][start:end]

def read_dataset():
	return None

# def rename(directory):
# 	for folder in os.listdir(directory):
# 		temp = os.path.join(directory,folder)
# 		if '_' in temp[-2:]:
# 			os.rename(temp,temp[:-1]+'0'+temp[-1])

header_names = ['youtube_id', 'vid_w', 'vid_h',
                'clip_start', 'clip_end', 'event_start', 'event_end',
                'event_start_ball_x', 'event_start_ball_y',
                'event', 'train_val_test']
actions = pd.read_csv(data_dir + action_path, header=0, names=header_names)
actions = actions.sort_values(['youtube_id', 'clip_start','event_end'])
videos_list = actions.youtube_id.unique()
train, val, test = split_train_val_test_index(257,42)
train_videos, val_videos, test_videos = videos_list[train], videos_list[val], videos_list[test]
train_dir = [os.path.join(processed_dir,video) for video in sorted(train_videos)]
a=get_clips(train_dir[:2])
print a