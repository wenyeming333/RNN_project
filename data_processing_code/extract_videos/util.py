def read_csv_data(action_path, bb_path):
	import pandas as pd

	header_names = ['youtube_id', 'vid_w', 'vid_h',
				'clip_start', 'clip_end', 'event_start', 'event_end',
				'event_start_ball_x', 'event_start_ball_y',
				'event', 'train_val_test']
	actions = pd.read_csv(action_path, header=0, names=header_names)
	# sort the actions dataframe accoriding to youtube_id and clip start time
	actions = actions.sort_values(['youtube_id', 'clip_start', 'event_start'])

	header_names_actions = ['youtube_id', 'time', 'x', 'y', 'w', 'h', 'id']
	bbs = pd.read_csv(bb_path, header=0, names=header_names_actions)
	bbs.time = bbs.time * 0.001 * 0.001# convert the time from microseconds to seconds

	return actions, bbs

def get_args():
	'''This function parses and return arguments passed in'''

	import argparse

	parser = argparse.ArgumentParser(        
		description='Script to preprocess NCAA data.')

	parser.add_argument('-v', '--video', type=str, help='video id',
		required=False, default='-KUYDYCwnOQ')

	parser.add_argument('-ch', '--create-html', action='store_true', help='create htmls', default=False)

	parser.add_argument('-d', '--draw-bbs', action='store_true', help='draw bbs', default=False)

	#parser.add_argument('-g', '--gpu', type=int, help='set gpu',
	#	required=True)

	args = parser.parse_args()
	video = args.video
	create_html = args.create_html
	draw_bbs = args.draw_bbs
	#gpu = args.gpu

	return video, create_html, draw_bbs

def setFileDirectory(root_directory, foldername):
	import os, shutil

	folder_path = root_directory + '/' + str(foldername)#+ title.replace(' ', '_')
	try:
		if not os.path.isdir(root_directory):
			print('Error: saving directory does not exist.')
			exit()
		if os.path.exists(folder_path):
			shutil.rmtree(folder_path)
		os.makedirs(folder_path)

		print('Directory \"{}\" made.'.format(folder_path))
	except Exception as e:
		print('Error in making the movie folder.')
		print(e)
		exit()

	return folder_path

