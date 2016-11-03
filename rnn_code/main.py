from variables import *
import time
import cPickle as Pickle
from util.util import *
from util.setData import RNNData
import numpy as np

def saveProcessAllData():
	from preprocess.inception_v3 import *
	from os import listdir
	from glob import glob 
	from os.path import isdir
	videos = [f for f in listdir(processed_dir) \
					if isdir('{}/{}'.format(processed_dir, f))]

	model = InceptionV3(include_top=False, weights='imagenet')
	
	for v in videos:
		clips = [f for f in listdir('{}/{}'.format(processed_dir, v))\
					if isdir('{}/{}/{}'.format(processed_dir, v, f))]
		video_path = setFileDirectory(features_dir, v)
		for clip in clips:
		
			clip_path = setFileDirectory(video_path, clip)
			start_time = time.time()
			images = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			
			clip_result = model.predict(images)
			
			print('Processing {}-{}:\n\tThe program takes {} seconds to run'.format(v, clip,
																	time.time()-start_time))
																	
			np.save('{}/frame_features.npy'.format(clip_path),clip_result)
				
			
			
def main():
	saveProcessAllData()
	
if __name__ == '__main__':
	#main()
	rnn_data = RNNData(data_dir, action_path, processed_dir)
	import pdb
	pdb.set_trace()