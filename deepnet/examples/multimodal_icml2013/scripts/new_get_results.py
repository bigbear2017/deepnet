import numpy as np
import sys, os
from deepnet import fx_util

NUM = 500

def test():
  data_dir = sys.argv[1]
  dic = fx_util.fx_unpickle(sys.argv[2])
  label_file = sys.argv[3]
  
  edge_image = dic['edge_image']
  edge_text = dic['edge_text']
  label_graph = dic['label_graph']
  
  image_reps = np.load(os.path.join(data_dir, 'train/image_restricted_hidden-00001-of-00001.npy'))
  text0_reps = np.load(os.path.join(data_dir, 'train/text_restricted_hidden-00001-of-00001.npy'))
  text1_reps = np.load(os.path.join(data_dir, 'validation/text_restricted_hidden-00001-of-00001.npy'))
  
  dist0 = ((image_reps - text0_reps)**2).sum(axis=1)
  dist1 = ((image_reps - text1_reps)**2).sum(axis=1)
  dist = np.c_[dist0, dist1]
  dist = dist / dist.sum(axis=1).reshape((NUM,1))
  res = np.zeros(NUM, dtype=np.int32) - 1
  
  def is_stop():
    for i in range(NUM):
      if len(edge_image[i]) != 0:
        return False
    return True
    
  def get_pair():
    for i in range(NUM):
      if len(edge_image[i]) == 1:
        return i, edge_image[i][0]
    maxv = -1
    a = -1
    b = -1
    for i in range(NUM):
      if len(edge_image[i]) == 2:
        if dist[i,0] > dist[i,1]:
          if maxv < dist[i,0] - dist[i,1]:
            maxv = dist[i,0] - dist[i,1]
            a = i
            b = edge_image[i][1]
        else:
          if maxv < dist[i,1] - dist[i,0]:
            maxv = dist[i,1] - dist[i,0]
            a = i
            b = edge_image[i][0]
    return a, b
  
  while(is_stop() == False):
    s, t = get_pair()
    res[s] = label_graph[s,t]
    edge_image[s].remove(t)
    edge_text[t].remove(s)
    if len(edge_text[t]) > 0:
      edge_image[edge_text[t][0]].remove(t)  
    if len(edge_image[s]) > 0:
      edge_text[edge_image[s][0]].remove(s)
    
    edge_text[t] = []
    edge_image[s] = []
  
  with open(label_file, 'r') as f:
    expected = np.array(f.readlines(), dtype='int32')
  
  print (res == expected).sum() / float(NUM)
    
if __name__ == '__main__':
  test()