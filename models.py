from keras.layers import Concatenate, Conv3D, Dropout, Input, MaxPooling3D, UpSampling3D,BatchNormalization,Activation,Cropping3D
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from variables import *

def iou_coef(y_true, y_pred, smooth=1):
#     y_true = tf.to_float(y_true);
#     y_pred = tf.to_float(y_pred>0.5);
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3,4])
    union = K.sum(y_true,[1,2,3,4])+K.sum(y_pred,[1,2,3,4])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
#     y_true = tf.to_float(y_true);
#     y_pred = tf.to_float(y_pred>0.5);
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def ConvolutionBlock(x, name, fms, params):
    x = Conv3D(filters=fms, **params, name=name+"_conv0")(x)
    x = BatchNormalization(name=name+"_bn0")(x)
    x = Activation("relu", name=name+"_relu0")(x)

    x = Conv3D(filters=fms, **params, name=name+"_conv1")(x)
    x = BatchNormalization(name=name+"_bn1")(x)
    x = Activation("relu", name=name)(x)
    return x

def dice_coef_loss(target, prediction, axis=(1, 2, 3, 4), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def create_model(edge_size,fms = 8, use_upsampling = True, input_shape = [None,None,None,1] ):

        inputs =  Input(shape=input_shape,name="inputs")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same",kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),padding="same")


        # BEGIN - Encoding path
        encodeA = ConvolutionBlock(inputs, "encodeA", fms, params)
        poolA =  MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ConvolutionBlock(poolA, "encodeB", fms*2, params)
        poolB =  MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ConvolutionBlock(poolB, "encodeC", fms*4, params)
        poolC =  MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ConvolutionBlock(poolC, "encodeD", fms*8, params)
#         poolD =  MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

#         encodeE = ConvolutionBlock(poolD, "encodeE", fms*16, params)
        # END - Encoding path

        # BEGIN - Decoding path
        if use_upsampling:
            up =  UpSampling3D(name="upE", size=(2, 2, 2) )(encodeD)
        else:
            up =  Conv3DTranspose(name="transconvE", filters=fms*8,**params_trans)(encodeD)
        
#         concatD = Concatenate(axis=-1, name="concatD")([up, encodeD])
#         decodeC = ConvolutionBlock(concatD, "decodeC", fms*8, params)

#         if use_upsampling:
#             up =  UpSampling3D(name="upC", size=(2, 2, 2) )(decodeC)
#         else:
#             up =  Conv3DTranspose(name="transconvC", filters=fms*4,**params_trans)(decodeC)
        
        concatC = Concatenate(axis=-1, name="concatC")([up, encodeC])
        decodeB = ConvolutionBlock(concatC, "decodeB", fms*4, params)

        if use_upsampling:
            up =  UpSampling3D(name="upB", size=(2, 2, 2) )(decodeB)
        else:
            up =  Conv3DTranspose(name="transconvB", filters=fms*2,**params_trans)(decodeB)
        
        concatB = Concatenate(axis=-1, name="concatB")([up, encodeB])
        decodeA = ConvolutionBlock(concatB, "decodeA", fms*2, params)

        if use_upsampling:
            up =  UpSampling3D(name="upA", size=(2, 2, 2) )(decodeA)
        else:
            up =  Conv3DTranspose(name="transconvA", filters=fms,**params_trans)(decodeA)
        concatA = Concatenate(axis=-1, name="concatA")([up, encodeA])

        # END - Decoding path
        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="valid",kernel_initializer="he_uniform")
        convOut = ConvolutionBlock(concatA, "convOut", fms, params)

        prediction =  Conv3D(name="PredictionMask",
                                     filters = 1, kernel_size=(1, 1, 1),
                                     activation="sigmoid")(convOut)
        croping_size = tuple([int(edge_size[i] - 2) for i in range(3)])
        prediction = Cropping3D(((croping_size[0],croping_size[0]),(croping_size[1],croping_size[1]),(croping_size[2],croping_size[2])))(prediction)

        model = Model(inputs=[inputs], outputs=[prediction])

        model.summary(positions=[.23, .55, .67, 1.])

        return model