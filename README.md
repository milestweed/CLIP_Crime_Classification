# :eye: Hindsight AI: Crime Classification With Clip

## About
**For Educational Purposes Only** This is a recursive neural net trained to classify specific crime classes based on the UCF-Crime dataset [UCF-CRIME](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) or to perform general anomaly detection.  The model uses images that have been encoded into the CLIP image embedding space.

## Introducing CLIP
The model we are utilizing in our application, CLIP (developed by OpenAI), is a generalized image classification model which can take any image and produce word embeddings for the purpose of matching raw text strings to the contents of the image. The design and training of the model allows for high zero-shot performance in classifying images (i.e. image classification problems outside of the training set). The following image provides a summary of the model (taken from A. Radford et al.):

<img src="images/clip.png?raw=true"/>

While typical image classification models train an image feature extractor and a linear classifier to predict a label, CLIP trains an image encoder and text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes.

## Installation

Clone the repo and the required packages can be found in the required.txt file.  Running classifier.py will start an interactive application that will attempt to perform anomaly detection or multi-class classification on videos found in the 'Videos' directory. 

The scripts that were used to create the image sequence database from the video files of the UCF-Crime dataset as well as the training scripts and models can be found in the src directory.
