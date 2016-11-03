from variables import *
from preprocess.inception_v3 import *
import time
import cPickle as Pickle
from util.util import *
import numpy as np

def saveProcessAllData():
	from os import listdir
	from glob import glob 
	
	videos = listdir(processed_dir)
	model = InceptionV3(include_top=False, weights='imagenet')
	
	for v in videos:
		clips = listdir('{}/{}'.format(processed_dir, v))
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
	main()