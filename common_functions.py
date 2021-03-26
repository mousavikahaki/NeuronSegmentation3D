"""
Created on Mon Aug 20 14:29:06 2018

@author: Shih-Luen Wang
"""
import numpy as np
from skimage import io
import os

def get_data(IM_path,label_path):

  IM = io.imread(IM_path).astype(float)
  IM = np.einsum('kij->ijk',IM)
  IM = (IM/255)
  label = io.imread(label_path).astype(float)
  label = np.einsum('kij->ijk',label)
  label = ((label==255)*0.5+(label != 0)*0.5)
  return IM, label

def max_proj(x):
  y = np.zeros((len(x),len(x[0])))
  for i in range(len(x)):
    for j in range(len(x[0])):
      y[i,j] = np.amax(x[i,j,:])
  return y
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def get_ind_IMlist(data_dir,N_images,out_x,out_y,out_z,pad_x,pad_y,pad_z):
    IM_name_list = []
    label_name_list = []
    for i in range(N_images):
        IM_name_list.append('image_'+str(i+1)+'.tif')
        label_name_list.append('label_'+str(i+1)+'.tif')
    Index_matrix = np.array([],dtype=int).reshape(0,4)
    train_IM_list = IM_name_list
    train_label_list = label_name_list
    for i in range(len(IM_name_list)):
      phantom_IM, phantom_label = get_data(data_dir+IM_name_list[i],data_dir+label_name_list[i])
      x_N = (phantom_IM.shape[0]-out_x+1)
      y_N = (phantom_IM.shape[1]-out_y+1)
      z_N = (phantom_IM.shape[2]-out_z+1)
      temp_Index_matrix = np.zeros([x_N*y_N*z_N,4],dtype=int)
      temp_Index_matrix[:,0] = i # which image
      temp_Index_matrix[:,1] = np.repeat(range(x_N),y_N*z_N) #x
      temp_Index_matrix[:,2] = np.tile(np.repeat(range(y_N),z_N),x_N)#y
      temp_Index_matrix[:,3] = np.tile(range(z_N),x_N*y_N)#z
      Index_matrix = np.concatenate((Index_matrix,temp_Index_matrix), axis=0)
      phantom_IM = np.pad(phantom_IM, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
      phantom_label = np.pad(phantom_label, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
      train_IM_list[i] = phantom_IM
      train_label_list[i] = phantom_label
    return Index_matrix, train_IM_list, train_label_list

def get_validation_set(data_dir,valid_IM_name,valid_label_name):
    valid_IM, valid_label = get_data(data_dir+valid_IM_name,data_dir+valid_label_name)      
    valid_IM = valid_IM[356:484,208:336,21:37]
    valid_label = valid_label[356:484,208:336,21:37]
    
    return valid_IM, valid_label