#!/bin/bash
# Before run this script, you must run the script 'runall_rbm.sh' first.

# Location of deepnet. EDIT this for your setup.
deepnet=$HOME/deepnet/deepnet

# Location of the downloaded data. This is also the place where learned models
# and representations extracted from them will be written. EDIT this for your setup.
prefix=$HOME/data/multimodal_icml2013

gpu_mem=4G
main_mem=20G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
extract_rep_nn=${deepnet}/extract_neural_net_representation.py
model_output_dir=${prefix}/restricted_models
data_output_dir=${prefix}/restricted_reps

models_dir=models/restricted
trainers_dir=trainers/restricted

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

if ${clobber} || [ ! -e ${data_output_dir}/restricted_layer_data ]; then
  echo "Generate data for restricted_layer."
  python scripts/generate_restricted_data.py \
    ${prefix}/rbm_reps/joint_rbm_LAST/input_data.pbtxt \
    ${data_output_dir}/restricted_layer_data/data.pbtxt \
    restricted_layer || exit 1
fi


if ${clobber} || [ ! -e ${model_output_dir}/restricted_layer_LAST ]; then
  echo "Training restricted_layer."
  python ${trainer} ${models_dir}/restricted_layer.pbtxt \
    ${trainers_dir}/train_CD_restricted_layer.pbtxt eval.pbtxt || exit 1
  
  cp ${model_output_dir}/restricted_layer_BEST ${model_output_dir}/restricted_layer_LAST
  
  echo "extract_image_reps_restricted_layer"
  python ${extract_rep_nn} ${models_dir}/extract_image_reps_restricted_layer.pbtxt \
      ${trainers_dir}/train_CD_restricted_layer.pbtxt ${data_output_dir}/restricted_layer_LAST \
      ${prefix}/private_test.pbtxt image_restricted_hidden || exit 1
      
  echo "extract_text_reps_restricted_layer"
  python ${extract_rep_nn} ${models_dir}/extract_text_reps_restricted_layer.pbtxt \
      ${trainers_dir}/train_CD_restricted_layer.pbtxt ${data_output_dir}/restricted_layer_LAST \
      ${prefix}/private_test.pbtxt text_restricted_hidden || exit 1
    
  python scripts/new_get_results.py ${data_output_dir}/restricted_layer_LAST \
      ${prefix}/private_test/forpredict.pkl \
      ${prefix}/private_test/labels.txt || exit 1
fi
