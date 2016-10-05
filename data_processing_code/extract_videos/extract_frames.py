import subprocess
import os



def extract_frames_command(video_path, output_dir, duration=-1):
    '''
    extract frames
    :param video_path:
    :param duration:
    :return:
    '''
    if duration == -1:
        command = "avconv -i " +  video_path + \
                  " -q:v 1 -qmax 1 " + output_dir + "%d.jpg"
    else:
        command = "avconv -i " + video_path + " -t " + str(duration) + \
                  " -q:v 1 -qmax 1 " + output_dir + "%d.jpg"

    return command


def extract_frames(processed_dir, video_name, video_path, duration, override=0):
    if override == 1:
        print("Overriding the previously extracted frames!")
    # extract the frames
    if not os.path.exists(processed_dir+video_name+ '/frames') or override == 1:
        print("Extracting Frames!")
        # no frames
        if not os.path.exists(processed_dir + video_name + '/frames'):
            os.mkdir(processed_dir+video_name + '/frames')

        comm = extract_frames_command(video_path, processed_dir+video_name + '/frames/', duration=duration)
        status = subprocess.call(comm, shell=True)
    else:
        print("Frames already extracted!")

