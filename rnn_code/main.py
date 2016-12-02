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

	videos.sort()
					
	model = load_model()
					
	load_features = load_get_output_fn(model, num_layer=217)
					
	for v in videos:
	
		print 'Processing video: {} ...'.format(v)
		clips = [f for f in listdir('{}/{}'.format(processed_dir, v))\
					if isdir('{}/{}/{}'.format(processed_dir, v, f))]

		clips.sort()
					
		video_path = '{}/{}'.format(features_dir, v)
		for clip in clips:
			im_dir, images = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			im_dir.sort()
			im_shape = images.shape
			
			#clip_features = {}
			clip_path = '{}/{}'.format(video_path, clip)
			#unique_ids = {}
			clip_features = []
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
				
				#clip_features[i] = {}
				frame_features = np.zeros([0, 2048], np.float32)
				for r_i in range(len(im_csv)):
					try:
						im_crop = np.swapaxes(im[:,y[r_i]:y_upper[r_i],
											x[r_i]:x_upper[r_i]], 0,2)
						
						im_crop = np.swapaxes(resize(im_crop, (299,299)),
												0,2)[np.newaxis,...]
							
						feature = np.reshape(load_features([im_crop, 0])[0], [1, -1])
						frame_features = np.concatenate((frame_features, feature))
						#clip_features[i][im_csv.id[r_i]] = feature.flatten()
						#if im_csv.id[r_i] not in unique_ids:
						#	feature_shape = feature.size
						#	unique_ids[im_csv.id[r_i]] = np.zeros((i, feature_shape))
						#unique_ids[im_csv.id[r_i]] = np.concatenate((unique_ids[im_csv.id[r_i]],
						#											feature.flatten()[np.newaxis,...]))
					except Exception as e:
						print e
						
				clip_features.append(frame_features)												
			#n_ids = len(unique_ids)
			
			#player_features = np.zeros((n_ids, images.shape[0], feature_shape))
			
			#for i, (p_id, features) in enumerate(unique_ids.iteritems()):
			#	player_features[i,:features.shape[0],:] = features 
				
			
			Pickle.dump(clip_features, open('{}/player_features.pkl'.format(clip_path), 'w'))
			#np.save('{}/player_features.npy'.format(clip_path),player_features)
			print ('\tFinish clip: {}'.format(clip))

def savePlayerSpatialFeatures():
	from preprocess.getPlayerFeatures import *
	
	from os import listdir
	from glob import glob 
	from os.path import isdir
	from skimage.transform import resize
	
	import pandas as pd
	
	videos = [f for f in listdir(processed_dir) \
					if isdir('{}/{}'.format(processed_dir, f))]

	videos.sort()
					
	model = load_model()
					
	load_features = load_get_output_fn(model, num_layer=217)
					
	for v in videos:
	
		print 'Processing video: {} ...'.format(v)
		clips = [f for f in listdir('{}/{}'.format(processed_dir, v))\
					if isdir('{}/{}/{}'.format(processed_dir, v, f))]

		clips.sort()
					
		video_path = '{}/{}'.format(features_dir, v)
		for clip in clips:
			im_dir, images = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			im_dir.sort()
			im_shape = images.shape
			
			#clip_features = {}
			clip_path = '{}/{}'.format(video_path, clip)
			#unique_ids = {}
			clip_features = []
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
				
				#clip_features[i] = {}
				frame_features = np.zeros([0, 200, 100], np.float32)
				for r_i in range(len(im_csv)):
					try:
						mask = np.zeros([im.shape[1], im.shape[2]])
						mask[y[r_i]:y_upper[r_i], x[r_i]:x_upper[r_i]] = 1
						mask = resize(mask, (200, 100))[np.newaxis,...]
						frame_features = np.concatenate((frame_features, mask))
					except Exception as e:
						print e
						
				clip_features.append(frame_features)												
			#n_ids = len(unique_ids)
			
			#player_features = np.zeros((n_ids, images.shape[0], feature_shape))
			
			#for i, (p_id, features) in enumerate(unique_ids.iteritems()):
			#	player_features[i,:features.shape[0],:] = features 
				
			
			Pickle.dump(clip_features, open('{}/spatial_features.pkl'.format(clip_path), 'w'))
			#np.save('{}/player_features.npy'.format(clip_path),player_features)
			print ('\tFinish clip: {}'.format(clip))
	
def main():
	#saveProcessAllData()
	#savePlayerFeatures()
	savePlayerSpatialFeatures()
	
if __name__ == '__main__':
	main()
	#rnn_data = RNNData(data_dir, action_path, processed_dir)
	import pdb
	pdb.set_trace()