import random
from deepnet import util
from deepnet import deepnet_pb2
import sys, os
from google.protobuf import text_format
import numpy as np
import glob
import copy

def generate_validation_data(proto, output_pbtxt):
  out_dir = '/'.join(output_pbtxt.split('/')[:-1])
  if out_dir and not os.path.isdir(out_dir):
    os.makedirs(out_dir)

  dataset = util.ReadData(proto)
  
  data_proto1 = next(d for d in dataset.data if d.name == 'image_hidden2_validation')
  data_proto2 = next(d for d in dataset.data if d.name == 'text_hidden2_validation')
  
  filename1 = sorted(glob.glob(os.path.join(dataset.prefix, data_proto1.file_pattern)))[0]
  filename2 = sorted(glob.glob(os.path.join(dataset.prefix, data_proto2.file_pattern)))[0]

  data1 = np.load(filename1)
  data2 = np.load(filename2)
  numcases, dimsions = data1.shape
  
  label_genuine = np.ones(numcases//2, dtype=int)
  label_impostor = np.zeros(numcases//2, dtype=int)

  label = np.concatenate((label_genuine, label_impostor))
  indices = np.arange(numcases)
  
  save_image_name = 'image_hidden2.npy'
  save_text_name = 'text_hidden2.npy'
  save_label_name = 'label_hidden2.npy'
  save_indices_name = 'indices_hidden2.npy'

   #validation
  data_dir = 'validation'
  image_name = 'image_hidden2_' + data_dir
  text_name = 'text_hidden2_' + data_dir
  label_name = 'label_hidden2_' + data_dir
  indices_name = 'indices_hidden2_' + data_dir
  
  dataset_pb = util.ReadData(output_pbtxt)
  
  full_data_dir = os.path.join(out_dir, data_dir)
  if full_data_dir and not os.path.isdir(full_data_dir):
    os.makedirs(full_data_dir)
  np.save(os.path.join(full_data_dir, save_image_name), data1)
  np.save(os.path.join(full_data_dir, save_text_name), data2)
  np.save(os.path.join(full_data_dir, save_label_name), label)
  np.save(os.path.join(full_data_dir, save_indices_name), indices)
  
  data_pb = dataset_pb.data.add()
  data_pb.name = image_name
  data_pb.file_pattern = data_dir + '/' + save_image_name
  data_pb.size = data1.shape[0]
  data_pb.dimensions.append(data1.shape[1])
  data_pb = dataset_pb.data.add()
  data_pb.name = text_name
  data_pb.file_pattern = data_dir + '/' + save_text_name
  data_pb.size = data2.shape[0]
  data_pb.dimensions.append(data2.shape[1])
  data_pb = dataset_pb.data.add()
  data_pb.name = label_name
  data_pb.file_pattern = data_dir + '/' + save_label_name
  data_pb.size = label.shape[0]
  data_pb.dimensions.append(1)
  data_pb = dataset_pb.data.add()
  data_pb.name = indices_name
  data_pb.file_pattern = data_dir + '/' + save_indices_name
  data_pb.size = indices.shape[0]
  data_pb.dimensions.append(1)
  
  with open(output_pbtxt, 'w') as f:
    text_format.PrintMessage(dataset_pb, f)
    
def main():
  proto = sys.argv[1]
  output_pbtxt = sys.argv[2]
  pb_name = sys.argv[3]
  
  out_dir = '/'.join(output_pbtxt.split('/')[:-1])
  if out_dir and not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  
  dataset_pb = deepnet_pb2.Dataset()
  dataset_pb.name = pb_name
  dataset_pb.gpu_memory = '4G'
  dataset_pb.main_memory = '30G'
  dataset_pb.prefix = out_dir
    
  dataset = util.ReadData(proto)

  data_proto1 = next(d for d in dataset.data if d.name == 'image_hidden2_train')
  data_proto2 = next(d for d in dataset.data if d.name == 'text_hidden2_train')
  
  filename1 = sorted(glob.glob(os.path.join(dataset.prefix, data_proto1.file_pattern)))[0]
  filename2 = sorted(glob.glob(os.path.join(dataset.prefix, data_proto2.file_pattern)))[0]

  data1 = np.load(filename1)
  data2 = np.load(filename2)
  
  numcases, dimsions = data1.shape
  assert numcases == 100000
  save_image_name = 'image_hidden2.npy'
  save_text_name = 'text_hidden2.npy'
  save_label_name = 'label_hidden2.npy'
  save_indices_name = 'indices_hidden2.npy'
    
  data_dir = 'train'
  image_name = 'image_hidden2_' + data_dir
  text_name = 'text_hidden2_' + data_dir
  label_name = 'label_hidden2_' + data_dir  
  indices_name = 'indices_hidden2_' + data_dir
  
  trainnum = 100000
  indices1 = np.arange(trainnum)
  indices2 = np.arange(trainnum)
  np.random.shuffle(indices2)
  eq_index = np.where(indices2 == indices1)[0]
  eq_num = len(eq_index)
  print eq_num, eq_index
  if eq_num == 1:
    first = indices2[eq_index[0]]
    indices2[eq_index[0]] = indices2[(eq_index[0]+1)%trainnum]
    indices2[(eq_index[0]+1)%trainnum] = first
  elif eq_num > 0:
    first = indices2[eq_index[0]]
    for i in range(eq_num - 1):
      indices2[eq_index[i]] = indices2[eq_index[i+1]]
    indices2[eq_index[eq_num-1]] = first
  assert (np.where(indices2 == indices1)[0]).sum() == 0
  label_genuine = np.ones(trainnum, dtype=int)
  label_impostor = np.zeros(trainnum, dtype=int)

  train_data1 = np.concatenate((data1,data1[indices1]))
  train_data2 = np.concatenate((data2,data2[indices2]))
  train_label = np.concatenate((label_genuine, label_impostor))  
  train_indices = np.arange(trainnum*2)
  
  full_data_dir = os.path.join(out_dir, data_dir)
  if full_data_dir and not os.path.isdir(full_data_dir):
    os.makedirs(full_data_dir)
  np.save(os.path.join(full_data_dir, save_image_name), train_data1)
  np.save(os.path.join(full_data_dir, save_text_name), train_data2)
  np.save(os.path.join(full_data_dir, save_label_name), train_label)
  np.save(os.path.join(full_data_dir, save_indices_name), train_indices)
  
  data_pb = dataset_pb.data.add()
  data_pb.name = image_name
  data_pb.file_pattern = data_dir + '/' + save_image_name
  data_pb.size = train_data1.shape[0]
  data_pb.dimensions.append(train_data1.shape[1])
  data_pb = dataset_pb.data.add()
  data_pb.name = text_name
  data_pb.file_pattern = data_dir + '/' + save_text_name
  data_pb.size = train_data2.shape[0]
  data_pb.dimensions.append(train_data2.shape[1])
  data_pb = dataset_pb.data.add()
  data_pb.name = label_name
  data_pb.file_pattern = data_dir + '/' + save_label_name
  data_pb.size = train_label.shape[0]
  data_pb.dimensions.append(1)
  data_pb = dataset_pb.data.add()
  data_pb.name = indices_name
  data_pb.file_pattern = data_dir + '/' + save_indices_name
  data_pb.size = train_indices.shape[0]
  data_pb.dimensions.append(1)
  
  print train_data1.shape, train_data2.shape, train_label.shape, train_indices.shape
        
  with open(output_pbtxt, 'w') as f:
    text_format.PrintMessage(dataset_pb, f)
  
  generate_validation_data(proto, output_pbtxt)
    
if __name__ == '__main__':
  main()