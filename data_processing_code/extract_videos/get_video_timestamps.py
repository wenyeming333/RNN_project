import subprocess
import os
import re
import pandas as pd

def get_timestamp_command(video_path, duration=-1):
    '''
    get the timestamps of the extracted frames
    :param video_path:
    :param duration:
    :return:
    '''
    if duration == -1:
        command = """avconv -i """ + video_path +  \
                  """ -filter:v "showinfo" -f null - """ + \
                  """2>&1 | grep 'pts_time' """
    else:
        command = """avconv -i """ + video_path + """ -t """+ str(duration) + \
                  """ -filter:v "showinfo" -f null - """ + \
                  """2>&1 | grep 'pts_time' """

    return command




def get_timestamps(processed_dir, video_name, video_path, duration, override=0):
    # given a video name, get the timestamps and write it to a file. But first check if the file
    # already exists or not. If it exists just load it

    if override == 1:
        print("Overriding the previously computed timestamps!")


    if not os.path.exists(processed_dir + video_name + '/timestamps.csv') or override == 1:
        print('Computing Timestamps!')

        comm = get_timestamp_command(video_path, duration=duration)

        # call the avconv to get the timestamps and store the output to a string
        timestamp_res = subprocess.check_output(comm, shell=True)
        # split the different lines
        timestamp_res = timestamp_res.splitlines()

        # timestamps is a list contaning the timestamps of each frame
        timestamps = []

        # get (parset) the timestamps
        for line in timestamp_res:
            time = re.findall(r'pts_time:[\d.]*', line)
            if len(time) > 0:
                time = time[0]
                timestamps.append(time[9:])

        timestamps = [float(ii)*1000 for ii in timestamps]
        # save the timestamps at processed_data/video_name/timestamps.csv
        df = pd.DataFrame(timestamps)
        df.to_csv(processed_dir + video_name + '/timestamps.csv', header=False, index=False)

        timestamps = pd.read_csv(processed_dir + video_name + '/timestamps.csv', header=0)

    else:
        print('Timestamps exist! Load them!')
        timestamps = pd.read_csv(processed_dir + video_name + '/timestamps.csv', header=0)

    return timestamps