from matplotlib import patches
from matplotlib import patheffects
from matplotlib import pyplot as plt
from variables import *
from utils import *
import numpy as np

def plot_max(Im,figsize=(8,8),ax = None,caxis=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(np.squeeze(np.amax(Im,2)),cmap='gray')
    if caxis is not None:
        ax.get_images()[0].set_clim(caxis)
    return ax

def draw_outline(o,lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw,foreground='black'),patheffects.Normal()])

def draw_rect(ax,b,edgecolor = 'red'):
    patch = ax.add_patch(patches.Rectangle((b[0],b[1]),b[2],b[3],fill=False,edgecolor=edgecolor,lw=1))
    draw_outline(patch,4)

def draw_text(ax,xy,txt,sz=8):
    text = ax.text(xy[0],xy[1],txt,verticalalignment = 'top',color='white',fontsize=sz,weight='bold')
    draw_outline(text,1)
    
def mark_individual_sample(Im_pred,input_size,stride,inds = None,figsize=(16,16),caxis = (0,0.2)):
    ax = plot_max(Im_pred,caxis = caxis,figsize=figsize)
    if inds is not None:
        for ind in inds:
            loc = ind2loc(Im_pred.shape,ind,input_size,stride)
            b = [loc[1],loc[0],input_size[0],input_size[0]]
            draw_rect(ax,b,edgecolor = 'red')
            draw_text(ax,b,str(ind))

def mark_high_error_region(Im_pred,input_size,stride,score,n = 10, mod = 'max',figsize=(16,16),caxis = (0,0.2)):
    if mod == 'max':
        score = -score
    ind = np.argsort(score)
    ax = plot_max(Im_pred,caxis = caxis,figsize=figsize)
    if n > 0:
        color = np.arange(0,1,1/n)
        for i in range(0,n):
            loc = ind2loc(Im_pred.shape,ind[i],input_size,stride)
            b = [loc[1],loc[0],input_size[0],input_size[0]]
            draw_rect(ax,b,edgecolor = [1,color[i],0])
            draw_text(ax,b,str(ind[i]))
            
def plot_history(history):
    plt.plot(history['binary_accuracy'])
    plt.plot(history['val_binary_accuracy'])
    plt.ylim([0.9996,1])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history['iou_coef'])
    plt.plot(history['val_iou_coef'])
    plt.title('Model iou_coef')
    plt.ylabel('iou_coef')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.ylim([0,0.005])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()