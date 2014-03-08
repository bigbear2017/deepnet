#!/bin/bash
# Location of deepnet. EDIT this for your setup.
HOME=/home/ec2-user/programs/
deepnet=$HOME/deepnet/deepnet

# Location of the downloaded data. This is also the place where learned models
# and representations extracted from them will be written. EDIT this for your setup.
prefix=$HOME/data/hackthon

gpu_mem=4G
main_mem=16G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/rbm_models
data_output_dir=${prefix}/rbm_reps

models_dir=models/rbm
trainers_dir=trainers/rbm

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

# Compute mean and variance of the data.
if [ ! -e  ${prefix}/imgstats.npz ]
then
  echo Computing mean / variance
  python ${deepnet}/compute_data_stats.py ${prefix}/espgame.pbtxt \
    ${prefix}/imgstats.npz train_data || exit 1
fi

# IMAGE LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm1_LAST ]; then
  echo "Training first layer image RBM."
  python ${trainer} ${models_dir}/image_rbm1.pbtxt \
    ${trainers_dir}/train_CD_image_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm1_LAST \
    ${trainers_dir}/train_CD_image_layer1.pbtxt image_hidden1 \
    ${data_output_dir}/image_rbm1_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi

# IMAGE LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm2_LAST ]; then
  echo "Training second layer image RBM."
  python ${trainer} ${models_dir}/image_rbm2.pbtxt \
    ${trainers_dir}/train_CD_image_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm2_LAST \
    ${trainers_dir}/train_CD_image_layer2.pbtxt image_hidden2 \
    ${data_output_dir}/image_rbm2_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi
