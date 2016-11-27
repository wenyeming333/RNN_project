import numpy as np
import cPickle
import time
import os

np.random.seed(111)

with open('/ais/gobi4/basketball/labels_dict.pkl') as f:
	labels_dict = cPickle.load(f)
with open('/ais/gobi4/basketball/olga_ethan_features/event_labels.pkl') as f:
	global_labels = cPickle.load(f)
with open('/ais/gobi4/basketball/current_videos_clips.pkl','rb') as f:
	current_videos_clips = np.array(sorted(cPickle.load(f)))

num_files = len(current_videos_clips)

print '================Loop starts================'
print 'there are {} files'.format(num_files)

frame_features = np.zeros([num_files, 20, 2048], dtype='float32')
player_features = np.zeros([num_files, 20, 10, 2048], dtype='float32')
labels = np.zeros([num_files, 11], dtype='float32')
seq_len = 20*np.ones([num_files])

start_time = time.time()

print '================Loop starts================'
print 'there are {} files'.format(num_files)

for j, clip_dir in enumerate(current_videos_clips):
	if j % 5 == 0: print j
	video_name, clip_id = clip_dir.split('/')[-2], clip_dir.split('/')[-1]
	class_name = global_labels[video_name][clip_id]
	labels[j, labels_dict.index(class_name)] = 1
	new_frame_features = np.load(os.path.join(clip_dir, 'frame_features.npy'))
	new_player_features = np.swapaxes(np.load(os.path.join(clip_dir, 'player_features.npy'))[:,:,:2048],0,1)
	num_frames = new_frame_features.shape[0]
	frame_features[j,:min(num_frames,20),:] = new_frame_features[:min(num_frames,20),:]
	num_player = new_player_features.shape[1]
	player_features[j,:min(num_frames,20),:min(num_player,10),:] = new_player_features[:min(num_frames,20),:min(num_player,10),:]

print 'it takes {} secs to finish'.format(time.time()-start_time)

features_dict = {}
features_dict['frame_features'] = frame_features
features_dict['player_features'] = player_features
features_dict['labels'] = labels
features_dict['seq_len'] = seq_len

with open('feature_dict.pkl','wb') as f:
	cPickle.dump(features_dict,f)