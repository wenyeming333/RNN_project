import numpy as np
import cPickle
import os


current_video = 'hahahaha'
current_num_videos = 125
features_dir = '/ais/gobi4/basketball/olga_ethan_features/'
videos = sorted(os.listdir(features_dir))

training_files = []
for i in range(current_num_videos):
	video_dir = os.path.join(features_dir, videos[i])
	clips = sorted(os.listdir(video_dir))
	for item in clips:
		training_files.append(os.path.join(video_dir, item))
training_files = sorted(training_files)

with open('current_videos_clips{}.pkl'.format(current_num_videos),'wb') as f:
	cPickle.dump(training_files,f)