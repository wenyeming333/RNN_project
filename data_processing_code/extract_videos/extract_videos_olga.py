import pandas as pd
import numpy as np
import os, shutil, glob
import re
import pdb

#import cv2 as cv
#import skvideo.io
from moviepy.editor import *

import random

import warnings
warnings.filterwarnings("ignore")

# load required local modules
from util import *

def main():
	# specify the paths to the data
	from variables import *

	# load data
	actions_data, bbs_data = read_csv_data(data_dir+action_path, data_dir+bb_path)

	# get arguments
	video_name, create_html, draw_bbs = get_args()

	if draw_bbs:
		import matplotlib.pyplot as plt
		plt.switch_backend('agg')
		import matplotlib.image as mpimg
		import matplotlib.patches as patches

	game_bbs = bbs_data[bbs_data.youtube_id == video_name]
	game_actions = actions_data[actions_data.youtube_id == video_name]
	game_actions = game_actions.reset_index()
	game_actions.event_end *= 0.001

	# get video
	video_path = data_dir + video_dir + video_name + '.mp4'
	video = VideoFileClip(video_path)

	# create a folder for the video where all the processed data will go to but first check if it exists
	processed_dir += 'olga/'
	if not os.path.exists(processed_dir):
		os.mkdir(processed_dir)
	video_dir = processed_dir + video_name
	if not os.path.exists(video_dir):
		os.mkdir(video_dir)

	# iterating over clips/events
	for clip_ind, clip in game_actions.iterrows():

		clip_dir = setFileDirectory(video_dir, 'clip_{}'.format(clip_ind+1))

		event_start = clip.event_end - 4
		clip_bbs = game_bbs[game_bbs.time >= event_start]
		clip_bbs = clip_bbs[clip_bbs.time <= clip.event_end]

		clip_bbs_uniq_times = clip_bbs.time.unique()
		count = 0

		color_assignment = {}
		captions = []

		with open('{}/clip_info.csv'.format(clip_dir), 'ab') as clip_info_f:
			clip.to_csv(clip_info_f)

		# iterating over frames
		for frame_time in clip_bbs_uniq_times:
			im_path = '{}/{:02}.jpg'.format(clip_dir, count)
			video.save_frame(im_path, t=frame_time)
			count += 1

			frame_bbs = clip_bbs[clip_bbs.time == frame_time]

			frame_bbs.x *= video.w
			frame_bbs.w *= video.w
			frame_bbs.y *= video.h
			frame_bbs.h *= video.h

			with open('{}/{:02}_info.csv'.format(clip_dir, count), 'ab') as bbs_f:
				frame_bbs.to_csv(bbs_f)

			# update caption information
			if create_html:
				cap = []
				cap.append('Num players detected: {}'.format(len(frame_bbs.x)))
				cap.append('Time: {}'.format(frame_time))
				captions.append(cap)

			# codes for drawing bbs
			if draw_bbs:
				im = mpimg.imread(im_path)
				fig = plt.figure(frameon=False)
				ax = plt.Axes(fig, [0., 0., 1., 1.])
				ax.set_axis_off()
				fig.add_axes(ax)

				ax.imshow(im)

				# iterating over bounding boxes
				for _, bbox in frame_bbs.iterrows():
					if bbox.id in color_assignment:
						color = color_assignment[bbox.id]
					else:
						color = (random.random(), random.random(), random.random())
						color_assignment[bbox.id] = color
							
					rect = patches.Rectangle((bbox.x, bbox.y),
											bbox.w, bbox.h, linewidth=3,
											edgecolor=color,
											facecolor='none')
					ax.add_patch(rect)

				dpi = fig.dpi
				fig.set_size_inches(video.w / dpi, video.h / dpi)
				fig.savefig(im_path)
				plt.close(fig)

		if create_html:
			from html_dashboard.codes.main import HTMLFramework
			# import error: add this to python path: https://github.com/oooolga/html_dashboard
			html = HTMLFramework('index', html_folder=clip_dir,
								 page_title='Files in {}'.format(clip_dir))

			images = glob.glob('{}/*.jpg'.format(clip_dir))

			images = [im[im.rfind('/')+1:] for im in images]
			images.sort()

			html.set_image_table(images, width=video.w, height=video.h, num_col=2, captions=captions,
							 sec_name='{}'.format(clip.event))

			html.write_html()

	if create_html:
		from html_dashboard.codes.list_directory import DirectoryHTML
		html = DirectoryHTML('{}'.format(video_dir))
		html.write_html()
		html = DirectoryHTML('{}'.format(processed_dir))
		html.write_html()
			

if __name__ == '__main__':
	main()