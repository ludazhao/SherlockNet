#!/bin/bash
echo "hello"
export CUDNN_PATH="/home/ubuntu/torch_3rd_party/cuda/lib64/libcudnn.so.5"

. /home/ubuntu/torch/install/bin/torch-activate


for i in $( ls /data/images_validation_10k_combined ); do
    th eval.lua -model /data/captioning/models/model_pretrained_coco.t7 -image_folder /data/images_validation_10k_combined/$i -output_folder $i -num_images -1
done