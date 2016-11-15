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
			_, images = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			
			clip_result = model.predict(images)
			
			print('Processing {}-{}:\n\tThe program takes {} seconds to run'.format(v, clip,
																	time.time()-start_time))
																	
			np.save('{}/frame_features.npy'.format(clip_path),clip_result)
				
def savePlayerFeatures():
	from preprocess.getPlayerFeatures import *
	
	from os import listdir
	from glob import glob 
	from os.path import isdir
	from skimage.transform import resize
	
	import pandas as pd
	
	videos = [f for f in listdir(processed_dir) \
					if isdir('{}/{}'.format(processed_dir, f))]
					
	model = load_model()
					
	load_features = load_get_output_fn(model)
					
	for v in videos:
	
		print 'Processing video: {} ...'.format(v)
		clips = [f for f in listdir('{}/{}'.format(processed_dir, v))\
					if isdir('{}/{}/{}'.format(processed_dir, v, f))]
					
		video_path = '{}/{}'.format(features_dir, v)
		for clip in clips:
			im_dir, images = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			im_dir.sort()
			im_shape = images.shape
			
			clip_features = {}
			clip_path = '{}/{}'.format(video_path, clip)
			for i in xrange(images.shape[0]):
				im = images[i,...]
				csv_path = im_dir[i][:im_dir[i].rfind('/')] + '/{:02}_info.csv'.format(i+1)
				im_csv = load_im_csv(csv_path)
				x = np.array(im_csv.x.values, np.int)
				y = np.array(im_csv.y.values, np.int)
				w = np.array(im_csv.w.values, np.int)
				h = np.array(im_csv.h.values, np.int)
				
				imshape = im.shape
				x[x<0] = 0
				x_upper = x+w 
				x_upper[x_upper>im_shape[-1]] = im_shape[-1]
				y[y<0] = 0
				y_upper = y+h
				y_upper[y_upper>im_shape[-2]] = im_shape[-2]
				
				clip_features[i] = {}
				for r_i in range(len(im_csv)):
					
					im_crop = np.swapaxes(im[:,y[r_i]:y_upper[r_i],
										x[r_i]:x_upper[r_i]], 0,2)
					
					try:
						im_crop = np.swapaxes(resize(im_crop, (299,299)),
											0,2)[np.newaxis,...]
						
						feature = load_features([im_crop, 0])[0]
						clip_features[i][im_csv.id[r_i]] = feature.flatten()
					except Exception as e:
						print e 
						pass
						
					  
			
			Pickle.dump(clip_features, open('{}/player_info.pkl'.format(clip_path), 'w'))
			print ('\tFinish clip: {}'.format(clip))
	
def main():
	#saveProcessAllData()
	savePlayerFeatures()
	
if __name__ == '__main__':
	main()
	#rnn_data = RNNData(data_dir, action_path, processed_dir)
	import pdb
	pdb.set_trace()