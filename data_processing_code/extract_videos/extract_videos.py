from __future__ import unicode_literals
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os
import re
import shutil
import glob
import pdb

from get_video_timestamps import get_timestamps
from extract_frames import  extract_frames
from savefigure import save_figure_as_image
from time import gmtime, strftime, ctime

# specify the paths to the data
from variables import *

if draw_bbs:
	import matplotlib.pyplot as plt
	plt.switch_backend('agg')
	import matplotlib.image as mpimg
	import matplotlib.patches as patches
	
colors = ['#b3ccff', '#ccff66', '#ff99cc', '#006699',
		  '#993366', '#ffff66', '#cc99ff', 'w', '0.5', '#ffe0cc',
		  '#669980', '#ff6666', '#cc3300', '#b3c6ff', '#ffcccc',
		  '#00cc00', '#003366', '#ff9999', '#666699', '#e0e0eb',
		  '#800000', '#999966', '#003300', '#00ccff', '#009933',
		  '#666699', 'r', 'g', 'c', 'm', 'y', 'b']

header_names = ['youtube_id', 'vid_w', 'vid_h',
				'clip_start', 'clip_end', 'event_start', 'event_end',
				'event_start_ball_x', 'event_start_ball_y',
				'event', 'train_val_test']
actions = pd.read_csv(data_dir + action_path, header=0, names=header_names)
# sort the actions dataframe accoriding to youtube_id and clip start time
actions = actions.sort_values(['youtube_id', 'clip_start', 'event_start'])

header_names_actions = ['youtube_id', 'time', 'x', 'y', 'w', 'h', 'id']
bbs = pd.read_csv(data_dir + bb_path, header=0, names=header_names_actions)
bbs.time = bbs.time * 0.001 # convert the time from microseconds to milliseseconds


#
# for video_name in actions.youtube_id.unique:
video_name = '-KUYDYCwnOQ'
# for what duration? if -1 the whole video
duration = -1
# create a folder for the video where all the processed data will go to but first check if it exists
if not os.path.exists(processed_dir+video_name):
	os.mkdir(processed_dir+video_name)

video_path = data_dir + video_dir + video_name + '.mp4'

# given a video name, get the timestamps and write it to a file. But first check if the file
# already exists or not. If it exists just load it
timestamps = get_timestamps(processed_dir, video_name, video_path, duration)
timestamps.columns = ['time']
timestamps.index += 1  # shift the index by one to correspond to the image name
# extract frames
#pdb.set_trace()
extract_frames(processed_dir, video_name, video_path, duration)

# now get the action data belonging to a a specific game
game_actions = actions[actions.youtube_id == video_name]
game_actions = game_actions.reset_index()
game_bbs = bbs[ bbs.youtube_id == video_name] #?????

game_bbs_uniq_times = game_bbs.time.unique()

game_wid = game_actions.vid_w[0]
game_h = game_actions.vid_h[0]


for clip_id, clip in game_actions.iterrows():

	print("Processing clip " + str(clip_id+1))

	# create a folder for the clip
	clip_dir = processed_dir + video_name + '/clip_' + str(clip_id+1)
	# create a directory for the clips
	if os.path.exists(clip_dir):
		shutil.rmtree(clip_dir)
	os.mkdir(clip_dir)

	event_end_ind = timestamps.index[(timestamps['time'] - clip.event_end).abs().argsort()[:1]].tolist()[0]
	event_start_ind = timestamps.index[(timestamps['time'] - (clip.event_end-4000)).abs().argsort()[:1]].tolist()[0]

	#for ii in range(event_start_ind, event_end_ind+1):
	#	shutil.copy(processed_dir+video_name+'/frames/'+str(ii)+'.jpg',
	#				clip_dir+'/'+str(ii)+'.jpg')
	
	color_assignment = {}
	count = 0

	# Now for each bounding box with a time that belongs to clip, add the bounding boxes to the image
	for time in game_bbs_uniq_times:
		
		if time >= (clip.event_end-4000) and time <= clip.event_end:

			img_bbs = game_bbs[game_bbs.time == time]

			#pdb.set_trace()

			ind = timestamps.index[(timestamps['time'] - time).abs().argsort()[:1]].tolist()[0]
			ind += 1 # since the image naming starts from 1

			img_name = clip_dir + '/' + str(ind) + '.jpg'
			shutil.copy(processed_dir+video_name+'/frames/'+str(ind)+'.jpg',
						img_name)

			img_bbs.is_copy = False

			img_bbs['bbs_x'] = pd.Series(np.array([bbox.x*game_wid for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)
			img_bbs['bbs_w'] = pd.Series(np.array([bbox.w*game_wid for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)
			img_bbs['bbs_y'] = pd.Series(np.array([bbox.y*game_h for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)
			img_bbs['bbs_h'] = pd.Series(np.array([bbox.h*game_h for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)

			img_bbs['im_dir'] = pd.Series(img_name, index=img_bbs.index)

			img_bbs['event'] = pd.Series(np.array([clip.event for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)
										 
			img_bbs['p_id'] = pd.Series(np.array(
										[re.findall(r'_\d*_', bbox.id)[0][1:-1] for _, bbox in img_bbs.iterrows()]),
										 index=img_bbs.index)
										 
			img_bbs['time'] = pd.Series(time, index=img_bbs.index)
										 
			with open('{}/{}_info.csv'.format(clip_dir, ind), 'ab') as info_f:
				img_bbs.to_csv(info_f)
			
			if draw_bbs:
				img = mpimg.imread(img_name)

				fig = plt.figure(frameon=False)
				ax = plt.Axes(fig, [0., 0., 1., 1.])
				ax.set_axis_off()
				fig.add_axes(ax)

				ax.imshow(img)
				
				for _, bbox in img_bbs.iterrows():
				
					if bbox.p_id in color_assignment:
						color = color_assignment[bbox.p_id]
					else:
						color = colors[count]
						count += 1
						color_assignment[bbox.p_id] = color
						
					rect = patches.Rectangle((bbox.bbs_x, bbox.bbs_y),
											 bbox.bbs_w, bbox.bbs_h, linewidth=3,
											 edgecolor=color,
											 facecolor='none')
					ax.add_patch(rect)

				dpi = fig.dpi
				fig.set_size_inches(game_wid / dpi, game_h / dpi)
				fig.savefig(img_name)
				plt.close(fig)

	try:
		from html_dashboard.codes.main import HTMLFramework
		html = HTMLFramework('index', html_folder=clip_dir,
							 page_title='Files in {}'.format(clip_dir))

		images = glob.glob('{}/*.jpg'.format(clip_dir))
		images.sort()
		captions = []
		cap_len = 4
		for im in images:
			info_file = '{}_info.csv'.format(im[:im.find('.jpg')])
			try:
				cap = []
				info_f = pd.read_csv(info_file)
				cap.append(info_f.event[0])
				cap.append('Num players detected: {}'.format(len(info_f.event)))
				cap.append('Time: {}'.format(info_f.time[0]))
				cap.append('Ind: {}'.format(im[im.rfind('/')+1:-4]))
				captions.append(cap)
			except:
				captions.append(['' for i in range(cap_len)])

		images = [im[im.rfind('/')+1:] for im in images]

		html.set_image_table(images, width=game_wid, height=game_h, captions=captions, num_col=2,
							 sec_name='Frame Images')

		info_files = glob.glob('{}/*.csv'.format(clip_dir))

		table = html.body.table()

		header_row = table.tr

		header_row.th().text('Name')
		header_row.th().text('Last Modified')
		header_row.th().text('Size Description')

		info_files = [f[f.rfind('/')+1:] for f in info_files]

		#pdb.set_trace()

		for f in info_files:
			row = table.tr
			row.td().a(href='{}'.format(f)).text('{}'.format(f))
			row.td(align='right').text('{}'.format(ctime(os.path.getmtime('{}/{}'.format(clip_dir, f)))))
			row.td(align='right').text('{}'.format(os.path.getsize('{}/{}'.format(clip_dir, f))))

		html.write_html()

	except Exception as e:
		print e

try:
	from html_dashboard.codes.list_directory import DirectoryHTML
	html = DirectoryHTML('{}{}'.format(processed_dir, video_name))
	html.write_html()
	html = DirectoryHTML('{}'.format(processed_dir))
	html.write_html()
except:
	pass