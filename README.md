# ArtHistoryNet

![Alt text](http://britishlibrary.typepad.co.uk/.a/6a00d8341c464853ef01a3fceb004b970b-500wi)

Using Convolutional Neural Networks to Explore Over 400 Years of Book Illustrations

# Set up
1. clone this to your local machine/GPU training server
2. set up a python virtualenv and install Tensorflow, following directions here: https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#virtualenv_install. 
Make sure to choose the right install package(for local test, use Mac-OSX with CPU, when install to server/gpu-enabled machine, use GPU install). 
3. Installing Kera https://github.com/fchollet/keras. Follow the instructions to configure TensorFlow as the backend for Kera. 
4. Run the cifar10_cnn.py script in models/ to train a small CNN to sanity-check. 

