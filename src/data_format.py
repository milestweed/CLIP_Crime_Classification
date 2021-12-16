import os
import cv2
import argparse
import threading
import joblib
from queue import Queue

'''
    Script to convert a video data set into an image sequence dataset. The videos of the original dataset should be separated into directories that indicate classes. The arguments that are required when calling the function are input (the base directory of the original dataset) and output (the desired location of the new image sequence dataset - this does not need to exist already).

    Optional arguments are resize-width (desired final width of frames), resize-height (desired final height of frames), num-threads (number of threads used for multi-threading processes), file-ext (file extension used for all video files)
'''

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help="Directory where input video files are located.")
ap.add_argument('-o', '--output', required=True, help="Location to save formatted dataset.")
ap.add_argument('-rw', '--resize-width', required=False, default=None, help="Desired width of frame output")
ap.add_argument('-rh', '--resize-height', required=False, default=None, help="Desired height of frame output")
ap.add_argument('-t', '--num-thread', required=False, default=30, help="Number of threads for multi-threading")
ap.add_argument('-e', '--file-ext', required=False, default='.mp4', help="Video file extension")
args = vars(ap.parse_args())


def frames_to_files(input_path, output_path, resize_shape=None):
    '''
        INPUTS:
            input_path   -> Path to video file to be split into frames.
            output_path  -> Path to directory to save frames
            resize_shape -> (optional) tuple describing output frame size (int,int)

        OUTPUT: returns the total number of frames written to the output path
    '''
    name_temp = "frame_{0:012d}.jpg"
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()

    count = 1
    while ret:
       # if resize_shape:
       #     cv2.resize(frame, resize_shape)
        cv2.imwrite(os.path.join(output_path, name_temp.format(count)), frame)
        ret, frame = cap.read()
        count += 1

    return count - 1


def process_thread(queue, input_dir, output_dir, cls, resize_shape=None, file_ext=".mp4"):
    class_lab = class_labs[cls]
    annot = ''
    while not queue.empty():
        filename = queue.get()
        vid_dir = filename.replace(file_ext, '')
        input_path = os.path.join(input_dir, cls, filename)
        output_path = os.path.join(output_dir, cls, vid_dir)
        try:
            os.mkdir(output_path)
        except FileExistsError:
            pass

        frames = frames_to_files(input_path, output_path, resize_shape)

        annot = annot + f"{os.path.join(cls,vid_dir)} 1 {frames} {class_lab}\n"
        queue.task_done()

    with open(os.path.join(output_dir, 'annotations.txt'),'a') as f:
        f.writelines(annot)


# Output dir and annotations.txt already exist at this point

def process_videos(input_dir, output_dir, cls, resize_shape=None, NUM_THREADS=30, file_ext=".mp4"):
    '''
        Generates the image sequences for all videos of one class and updates the annotations.txt file
        INPUT:
            input_dir  -> str - Directory containing videos separated into class directories
            output_dir -> str - Desired destination for final image sequence dataset
            cls        -> str -  Name of class names (should correspond to directories of input_dir)
            resize_shape -> (int,int) - (Optional) Tuple indicating the final size of all frames (height, width)
            NUM_THREADS-> int - Sets the number of threads for multi-threading {Default:30}
            file_ext   -> str - the file extention for the video files {Default:30}

        OUTPUT: None
    '''

    class_in_dir = os.path.join(input_dir, cls)
    class_out_dir = os.path.join(output_dir,cls)
    try:
        os.mkdir(class_out_dir)
    except FileExistsError:
        print(f"{cls} class directory exists...\nContinuing...")

    video_filenames = os.listdir(class_in_dir)

    queue = Queue()
    [queue.put(filename) for filename in video_filenames]

    print(f'Generating image sequences for the {cls} class')
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=process_thread, args=(queue, input_dir, output_dir, cls, resize_shape, file_ext))
        worker.start()
    print(f'{queue.qsize()} videos to be completed')
    queue.join()
    print(f'{cls} class complete.')

if __name__ == '__main__':

    input_dir = args['input']
    output_dir = args['output']
    NUM_THREADS = args['num_thread']
    resize_width = args['resize_width']
    resize_height = args['resize_height']
    file_ext = args['file_ext']


    resize_shape = (resize_height, resize_width)
    # Try to create location if it does not already exist
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("Output directory exists...\nContinuing...")

    # Class names from Input directory
    class_list = os.listdir(input_dir)
    # Dictionary encoding class labels
    class_labs = {c:x for x,c in enumerate(class_list)}
    joblib.dump(class_labs, os.path.join(output_dir,"class_labs.pkl"))

    for cls in class_list:
        process_videos(input_dir, output_dir, cls, resize_shape, NUM_THREADS, file_ext)

    print('All videos processed')
