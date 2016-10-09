from __future__ import unicode_literals
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pdb

from get_video_timestamps import get_timestamps
from extract_frames import  extract_frames
from savefigure import save_figure_as_image

# specify the paths to the data
from variables import *

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

#temp.ix[(temp['clip_start'] - 518685.0).abs().argsort()[:2]]

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
#timestamps.index += 1  # shift the index by one to correspond to the image name
# extract frames
#pdb.set_trace()
#extract_frames(processed_dir, video_name, video_path, duration)

# now get the action data belonging to a a specific game
game_actions = actions[actions.youtube_id == video_name]
game_actions = game_actions.reset_index()
game_bbs = bbs[ bbs.youtube_id == video_name] #?????

game_bbs_uniq_times = game_bbs.time.unique()

game_wid = game_actions.vid_w[0]
game_h = game_actions.vid_h[0]


for clip_id, clip in game_actions.iterrows():
    #clip_id = 0
    #clip = game_actions.ix[0]

    print("Processing clip " + str(clip_id+1))

    # create a folder for the clip
    clip_dir = processed_dir + video_name + '/clip_' + str(clip_id+1)
    # create a directory for the clips
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

    # create a directory for the frames of that clip
    if not os.path.exists(clip_dir + '/frames'):
        os.mkdir(clip_dir + '/frames')

    if not os.path.exists(clip_dir + '/events')
        os.mkdir(clip_dir + '/events')

    # copy all the frames belonging to this clip tp the clip_dir+/frames directory
    clip_start_ind = timestamps.index[(timestamps['time'] - clip.clip_start).abs().argsort()[:1]].tolist()[0]
    clip_end_ind = timestamps.index[(timestamps['time'] - clip.clip_end).abs().argsort()[:1]].tolist()[0]

    for ii in range(clip_start_ind, clip_end_ind+1):
        shutil.copy(processed_dir+video_name+'/frames/'+str(ii)+'.jpg',
                    clip_dir+'/frames/'+str(ii)+'.jpg')

    event_end_ind = timestamps.index[(timestamps['time'] - clip.event_end).abs().argsort()[:1]].tolist()[0]
    event_start_ind = timestamps.index[(timestamps['time'] - (clip.event_end-4000)).abs().argsort()[:1]].tolist()[0]

    # Now for each bounding box with a time that belongs to clip, add the bounding boxes to the image
    for time in game_bbs_uniq_times:
        print('clip: '+ str(clip_id+1)+' ,time: '+ str(time))
        if time >= clip.clip_start and time <= clip.clip_end:

            #time = game_bbs_uniq_times[0]

            img_bbs = game_bbs[game_bbs.time == time]

            #time  = 518684.833

            ind = timestamps.index[(timestamps['time'] - time).abs().argsort()[:1]].tolist()[0]
            ind += 1 # since the image naming starts from 1

            #timestamps.ix[(timestamps['time'] - time).abs().argsort()[:2]]

            img_name = clip_dir + '/frames/' + str(ind) + '.jpg'

            img = mpimg.imread(img_name)

            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(img)

            for tmpidx, bbox in img_bbs.iterrows():

                bbox.x *= game_wid
                bbox.w *= game_wid
                bbox.y *= game_h
                bbox.h *= game_h

                rect = patches.Rectangle((bbox.x, bbox.y), bbox.w, bbox.h, linewidth=4, edgecolor='w', \
                                         facecolor='none')
                ax.add_patch(rect)

                # get the player id
                player_id = re.findall(r'_\d*_', bbox.id)[0][1:-1]
                ax.annotate(player_id, (bbox.x, bbox.y+70), color='w', fontsize=36)

            dpi = fig.dpi
            fig.set_size_inches(game_wid / dpi, game_h / dpi)

            fig.savefig(img_name)
            plt.close(fig)

    ## next step is to plot the action name on the image between clip.event_start and clip.event_end
    if clip.event_start != -1 and clip.event_start >= clip.clip_start:
        event_start_ind = timestamps.index[(timestamps['time'] - clip.event_start).abs().argsort()[:1]].tolist()[0]
    else:
        event_start_ind = timestamps.index[(timestamps['time'] - (clip.event_end-4000)).abs().argsort()[:1]].tolist()[0]

    event_end_ind = timestamps.index[(timestamps['time'] - clip.event_end).abs().argsort()[:1]].tolist()[0]

    for ii in range(event_start_ind, event_end_ind + 1):
        if ii > 0:
            #ii = event_start_ind
            img_name = clip_dir + '/frames/' + str(ii) + '.jpg'

            img = mpimg.imread(img_name)

            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(img)

            ax.text(20, 20, clip.event,
                            bbox={'facecolor':'red', 'pad':10})

            dpi = fig.dpi
            fig.set_size_inches(game_wid / dpi, game_h / dpi)

            fig.savefig(img_name)
            plt.close(fig)

