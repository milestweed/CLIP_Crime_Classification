# :eye: Hindsight AI: Crime Classification With Clip

## Introducing CLIP
The model we are utilizing in our application, CLIP (developed by OpenAI), is a generalized image classification model which can take any image and produce word embeddings for the purpose of matching raw text strings to the contents of the image. The design and training of the model allows for high zero-shot performance in classifying images (i.e. image classification problems outside of the training set). The following image provides a summary of the model (taken from A. Radford et al.):

<img src="images/clip.png?raw=true"/> 

While typical image classification models train an image feature extractor and a linear classifier to predict a label, CLIP trains an image encoder and text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes.

The approach we are taking in utilizing CLIP in our application is running equally spaced frames of a video through the model, producing word embeddings for the entire video. We then take user input in the form of a text string query, calculate probabilities of matches between text and word embeddings for each of the frames, and subsequently deliver frames/timestamps of interest based on the user query. As part of our pipeline, we are also implementing frame segmentation to allow the user to adjust the level of clarity of each query. This takes the form of a custom function which takes in image tensors for frames of the video along with an argument for the level of clarity, and outputs smaller overlapping sub-images/tensors. The sub-images will then be fed to CLIP, word embeddings will be compared to text strings, and the highest probability match will be outputted.  
