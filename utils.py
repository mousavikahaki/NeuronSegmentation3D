import matplotlib.pyplot as plt
import numpy as np
from variables import *
from models import *
from datas import *


def ind2loc(Im_shape,index,input_size,stride):
    stride_size = tuple([int((Im_shape[i] - input_size[i])/stride[i] + 1) for i in range(3)])
    i,j,k = np.unravel_index(index, stride_size, 'F')
    return (i*stride[0],j*stride[1],k*stride[2])

def get_nb_index(Im_shape,index,input_size,stride):
    stride_size = tuple([int((Im_shape[i] - input_size[i])/stride[i] + 1) for i in range(3)])
    nb_index = [index-1,index+1,index-stride_size[0],index+stride_size[0],index-stride_size[0]*stride_size[1],index+stride_size[0]*stride_size[1]]
    print(f'up:{nb_index[0]},down:{nb_index[1]},left:{nb_index[2]},right:{nb_index[3]},above:{nb_index[4]},below:{nb_index[5]}')
    
def str_time():
    import datetime

    x = datetime.datetime.now()
    strtime = x.strftime("%m/%d_%H:%M")
    
    return strtime
    
def train_val_test_split():
    training_data = ['20190905Worm1MaxStack_Manual_trace_label.mat',
                '20200204-3DScannedCG118B-2_Manual_trace_label.mat',
                '20200222-3DScanCG118B-3_Manual_trace_label.mat',
                '20200223-3DScanCG118B-2_Manual_trace_label.mat',
                '20200223-3DScanCG118B-3_Manual_trace_label.mat',
                '20200223-3DScanCG118B-4_Manual_trace_label.mat']
    
    test_data = ['20200204-3DScannedCG118B-3_Manual_trace_label.mat']
    return training_data,test_data

def get_individual_iou_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2,3,4))
    union = np.sum(y_true,(1,2,3,4))+np.sum(y_pred,(1,2,3,4))-intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def get_individual_binary_crossentropy(y_true, y_pred):
    result = []
    epsilon = 0.000000001
    y_pred = np.vectorize(lambda x:max(min(x, 1 - epsilon), epsilon))(y_pred)
    for i in range(y_pred.shape[0]):
        result.append(-np.mean(y_true[i,:]*np.log(y_pred[i,:]) + (1 - y_true[i,:]) * np.log(1 - y_pred[i,:])))
    return np.array(result)

def load_model(filename):
    dependencies = {
    'iou_coef': iou_coef,'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss
    }

    from keras.models import load_model 
    model = load_model(weights_path + filename,custom_objects=dependencies)
    return model

def model_predict(model,filenames,input_size,edge_size,stride,TTA = True):
    images = []
    pred = []
    for i,filename in enumerate(filenames):
        test_image,_ = load_mat_file(filename,input_size,edge_size,stride)
        test_output = App3DUNetIm_test(model,test_image,input_size,edge_size,stride,TTA = TTA)
        images.append(test_image)
        pred.append(test_output)
    return images,pred

def test_time_augmentation(model,x):
    if x.ndim == 4:
        x = x[np.newaxis,:,:,:,:]

    original = lambda x:x
    fliplr = lambda x: np.flip(x,1)
    flipud = lambda x: np.flip(x,2)
    flipz = lambda x: np.flip(x,3)
    rot90 = lambda x: np.rot90(x,1,(1,2))
    rot180 = lambda x: np.rot90(x,2,(1,2))
    rot270 = lambda x: np.rot90(x,3,(1,2))
    rot_back_90 = lambda x: np.rot90(x,-1,(1,2))
    rot_back_180 = lambda x: np.rot90(x,-2,(1,2))
    rot_back_270 = lambda x: np.rot90(x,-3,(1,2))

    augmentations = [original,fliplr,flipud,flipz,rot90,rot180,rot270]
    augmentations_back = [original,fliplr,flipud,flipz,rot_back_90,rot_back_180,rot_back_270]

    x_augs = [aug(x) for aug in augmentations]

    x_augs_pred = [model.predict(x) for x in x_augs]

    for i in range(len(x_augs_pred)):
        x_augs_pred[i] = augmentations_back[i](x_augs_pred[i])
        x_augs_pred[i] = x_augs_pred[i][np.newaxis,:,:,:,:,:]
        
    y = np.concatenate(x_augs_pred)

    y = np.mean(y,axis=0)
    return y

def App3DUNetIm_test(model,Im,input_size,edge_size,stride,TTA = True):  
    x, _ = Im2Data(Im,input_size,edge_size,stride)
    masks = create_circular_mask(x.shape[:-1])
    x = masks*x
    if TTA: 
        y = test_time_augmentation(model,x)
    else:
        y = model.predict(x)
    
    output_size = tuple([int(input_size[i] - 2*edge_size[i]) for i in range(3)])
    stride_size = tuple([int((Im.shape[i] - input_size[i])/stride[i] + 1) for i in range(3)])
    Im_pred = np.zeros(Im.shape)

    l = 0
    for k in range(stride_size[2]):
        for j in range(stride_size[1]):
            for i in range(stride_size[0]):
                pos_i,pos_j,pos_k = i*stride[0],j*stride[1],k*stride[2]
                temp_IM = y[l,:,:,:,0]
                SM = Im_pred[pos_i+edge_size[0]:pos_i+edge_size[0]+output_size[0],pos_j+edge_size[1]:pos_j+edge_size[1]+output_size[1],pos_k+edge_size[2]:pos_k+edge_size[2]+output_size[2]]
                Im_pred[pos_i+edge_size[0]:pos_i+edge_size[0]+output_size[0],pos_j+edge_size[1]:pos_j+edge_size[1]+output_size[1],pos_k+edge_size[2]:pos_k+edge_size[2]+output_size[2]] = \
                np.maximum(SM,temp_IM)
                l += 1
    return Im_pred