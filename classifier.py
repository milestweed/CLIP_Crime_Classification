import os
from time import sleep

from utils.analyzer import classify

def main():
    # clear screen to start
    os.system('cls' if os.name == 'nt' else 'clear')

    # Title page
    print('#'*40)
    print('#' + ' '*38 +'#')
    print('# Welcome to the CLIP Crime Classifier #')
    print('#' + ' '*38 +'#')
    print('#'*40)

    sleep(2)

    os.system('cls' if os.name == 'nt' else 'clear')

    # Get classification method
    print('Please choose a classification method\n1) Anomaly\n2) Multi-Class')
    cls_type = int(input('Selection number: '))

    os.system('cls' if os.name == 'nt' else 'clear')

    # File paths
    root = str(os.getcwd())
    video_path = os.path.join(root, 'Videos')

    # Collect video file options
    videos = os.listdir(video_path)

    #  Request Video to analyze
    print('Please select a video to classify:')
    for i, vid in enumerate(videos):
        print(f"{i+1}) {vid}")

    vid_choice = int(input('Selection number: ')) - 1

    selected_video_path = os.path.join(video_path, videos[vid_choice])

    classify(selected_video_path, videos[vid_choice], cls_type)

if __name__ == '__main__':

    main()
