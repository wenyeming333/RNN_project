import numpy as np
import pandas as pd

class RNNData:
	def __init__(self, data_dir, action_path, features_dir, seed=111,
				 split_files={'train':212,'valid':12,'test':33}):
		from os import listdir
		from glob import glob 
		from os.path import isdir
		self.data_dir = data_dir
		self.action_path = action_path
		self.features_dir = features_dir
		self.seed = seed
		self.split_files = split_files
		
		self.features_dir = features_dir
		self.videos = [f for f in listdir(self.features_dir)\
						if isdir('{}/{}'.format(self.features_dir, f))]
		train,val,test = self._split_train_val_test_index(len(self.videos))
		
		videos = np.array(self.videos)
		train_videos, valid_videos, test_videos = \
			videos[train], videos[val], videos[test]
			
		self.videos = {'videos': videos,
					   'train': train_videos,
					   'valid': valid_videos,
					   'test': test_videos}
					   
		self.clips = {}
		self.set_clips()
		
		
	def _split_train_val_test_index(self, total_num):
		from sklearn.cross_validation import train_test_split
		total = np.arange(total_num)
		train,temp = train_test_split(total,train_size=self.split_files['train'],
									  random_state=self.seed)
		val,test = train_test_split(temp,test_size=self.split_files['test'],
									random_state=self.seed)
		return train,val,test
		
		
	def set_clips(self):
	
		from os import listdir
		from glob import glob 
		from os.path import isdir
		modes = ['train', 'valid', 'test']
		self.num_clip = {}
		for mode in modes:
			self.clips[mode] = []
			for v in self.videos[mode]:
				self.clips[mode] += ['{}/{}'.format(v, f) for f in listdir('{}/{}'.format(self.features_dir, v)) \
										if isdir('{}/{}/{}'.format(self.features_dir, v, f))]
			self.num_clip[mode] = len(self.clips[mode])
			
	
	def next_batch_generator(self, batch_size, mode='train'):
	
		assert batch_size <= self.num_clip[mode]
		perm = np.arange(self.num_clip[mode])
		while True:
			ind = np.random.choice(perm, batch_size, replace=False)
			yield np.array(self.clips[mode])[ind]