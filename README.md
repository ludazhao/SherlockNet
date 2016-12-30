# SherlockNet
## Using Convolutional Neural Networks to Explore Over 400 Years of Book Illustrations

![Alt text](http://britishlibrary.typepad.co.uk/.a/6a00d8341c464853ef01a3fceb004b970b-500wi)

Starting from February 2016, as part of the [British Library Labs Competition](http://labs.bl.uk/British+Library+Labs+Competition), we embarked on a collaboration with the British Library Labs and the British Museum to tag and caption the entire British Library 1M Collection, a set of 1 million book illustrations scanned from books published between 1500 and 1900. We proposed to use convolutional neural networks (CNNs) to perform this automatic tagging and captioning. In addition, we proposed deeper analysis of temporal trends in these images using the explanatory power provided by neural networks. Below we provide our deliverables as well as short explanations of the iPython notebooks we wrote for our project. 

Our tags and captions can be found at our web portal [here](bit.ly/sherlocknet). We have also uploaded all our tags to Flickr [here](https://www.flickr.com/photos/britishlibrary/).

Disclaimer: This code is research quality only and should be treated as such. 

## Writeups and Slides

1. [Poster for Stanford CS231N class (March '16)](https://drive.google.com/open?id=0By39R6hglwcDVWNwNkt6blgtdms)
2. [Writeup for Stanford CS231N class (March '16)](https://drive.google.com/open?id=0By39R6hglwcDRUJVb0U3Ulo3cTA)
3. [Supplemental Figures (March '16)](https://drive.google.com/open?id=0By39R6hglwcDNEdvZFFEal8xZnc)
4. [Proposal for British Library funding (April '16)](http://labs.bl.uk/SherlockNet)
5. [Announcement of Finalist status (May '16)](http://blogs.bl.uk/digital-scholarship/2016/06/announcing-the-bl-labs-competition-finalists-for-2016.html)
6. [Progress Notes #1 (June '16)](http://blogs.bl.uk/digital-scholarship/2016/08/sherlocknet-tagging-and-captioning-the-british-librarys-flickr-images.html)
7. [Progress Presentation (Sep '16)](https://drive.google.com/open?id=0BxI6DIzmhgSBOG5RbzBWUF9Yc2s)
8. [Progress Notes #2 (Sep '16)](http://blogs.bl.uk/digital-scholarship/2016/11/sherlocknet-update-millions-of-tags-and-thousands-of-captions-added-to-the-bl-flickr-images.html)
9. [Final Presentation Slides (Nov '16)](https://drive.google.com/open?id=0BxI6DIzmhgSBQlZhamhxNDE0cDBSMElQMHdfNnM4WURzSkJz)
10. [Final Reflections (Dec '16)](https://docs.google.com/document/d/1pU1eN23oZvu9ffEhYhShLOVkzNPp7DZDn9P8qbUGnOc/edit?usp=sharing)

## Key pieces of code

#### 1. Data Preprocessing

* [preprocess.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/preprocess_all.py): This makes all images 256x256 and grayscale. It scales all images such that the 'smaller' dimension is 256 pixels, then crops the image to a square.
* [augment_data.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/augment_data.py) and [image_util.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/image_util.py): This augments our training set with rotations, crops, and other transformations to increase robustness.

#### 2. Training and Tagging

* [retrain.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/retrain.py): Modified TensorFlow for our needs; we performed 2 training steps, first on a manually classified 1.5K image training set, then on a machine-classified and manually validated 10K training set.
* [tag_analysis_on_manual_tags.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/tag_analysis_on_manual_tags.ipynb): We analyzed how well our 1.5K training set was classified by our model.
* [tag_analysis_on_10K_tags.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/tag_analysis_on_10k_tags.ipynb): We analyzed how well this 1.5K model performed on a larger 10K test set.
* [tag_1M.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/tag_1M.py): Tagging all 1M images using the new 10K model.
* [tags_net_analysis.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/tags_net_analysis.ipynb): Loss functions as our model was training.

#### 3. Analysis of Tags and Trends

* [1M Tag Analysis.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/analysis/1M%20Tag%20Analysis.ipynb): Gathering statistics for our dataset -- how many images from each tag, and trends over time
* [look_at_dual_tags.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/analysis/look_at_dual_tags.ipynb): How many images have two almost equally likely tags, and what does that say about the images?
* [analyze_maps_by_decade.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/analyze_maps_by_decade.ipynb): We trained a new model to categorize maps into 4 eras, then analyzed which neurons in this model became more or less active over time. To understand what each neuron represented, we found images that either highly activated the neuron or did not activate it at all.
* [analyze_decorations_by_decade.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/analyze_decorations_by_decade.ipynb): Similar analysis but with decorations.
* [retrain_decades.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts_decadesnet/retrain_decades.py): Script used by the above 2 analyses to retrain model to categorize images into eras.

#### 4. Obtaining text around each image, and generating tags using a nearest neighbor voting process

* [get_json_text.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/get_json_text.py): Save the text from the 3 pages around each image.
* [extract_noun_phrases.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/extract_noun_phrases.py): For each image, extract all noun phrases from its surrounding OCR text..
* [cluster_images_by_ocr.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/cluster_images_by_ocr.ipynb): Exploring ways to cluster images into topics using Tf-idf, PCA, and LDA
* [nearest_neighbor_ocr_tagging_pca.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/nearest_neighbor_ocr_tagging_pca.ipynb): Perform PCA within each category using 10K random images chosen from that category. Transform each image's 2048-dimensional representation from the CNN into a 40-dimensional vector, and calculate the 20 most similar images for each image in this 40D space.
* [get_final_tags.py](https://github.com/ludazhao/SherlockNet/blob/master/scripts/get_final_tags.py): For each image, have the 20 most similar images vote on the 20 words that appear most often in all of the images. Before this voting process, the script also performs spell check and stemming on the words.
* [filter_final_tags.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/filter_final_tags.ipynb): For each image, make sure its tags are all correctly spelled, and take out stopwords.

#### 5. training and generating captions

Note: This part leverages the open-source package [neuraltalk2](https://github.com/karpathy/neuraltalk2) for training and evaluating iamge captions, with slight modifications. Our generous thanks to its author, Andrej Karpathy.

* [prepro.py]() Converts a json files of captions to images into an form easily feed-able into Torch.
* [train.lua]() training script. For the list of hyperparameters, see the top of the file. 
* [eval.lua, ]() run the evaluation script to generate captions in the /vis folder. See [eval_10k.sh]() for an example of usage details.

For more usage details, please also consult the documentation of [neuraltalk2](https://github.com/karpathy/neuraltalk2). 

#### 6. Preprocessing, training, tagging & captioning experiments for the British Museum Prints and Drawings(BM) dataset

* The notebooks are named in order. Please consult in-line headers for more details on each notebook. 

#### 7. Uploading tags, captions to Flickr

* [image_name_to_url.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/image_name_to_url.ipynb): From the British Library documentation, find the flickr ID for each image.
* [upload_to_flickr.ipynb](https://github.com/ludazhao/SherlockNet/blob/master/scripts/upload_to_flickr.ipynb): Add tags to the Flickr page for each image with a "sherlocknet:" prefix.

## Data
We will publish our data, both in its raw form and its processed form, at a separate portal. Details coming soon!
