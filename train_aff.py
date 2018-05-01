import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
import time
from keras.optimizers import SGD, Adam
from keras.callbacks import *
from keras.objectives import *
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from train import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
# from tf_image_segmentation.recipes.mscoco import data_coco


if __name__ == '__main__':
    dataset = 'IIT-AFF'
    CONFIG = 'EXP' # Choose pre-configured set ('Pascal' -> train.py, 'COCO' -> train_coco.py)

    ###########################################

    if CONFIG == 'EXP':
        model_name = 'AtrousFCN_Resnet50_Aff'
        if True: # for test
            input_shape = (512,512,3)
            weight_decay = 1e-4
            batchnorm_momentum = 0.95
            print('1')
            seg_classes = 21
            aff_classes = 10
            model = globals()[model_name](weight_decay=weight_decay,
                                  input_shape=input_shape,
                                  batch_momentum=batchnorm_momentum,
                                  seg_classes=seg_classes,
                                  aff_classes=aff_classes)
            print('2')
            model.summary()
            exit(0)
        # model_name = 'Atrous_DenseNet'
        # model_name = 'DenseNet_FCN'
        batch_size = 16
        batchnorm_momentum = 0.95
        epochs = 250
        lr_base = 0.01 * (float(batch_size) / 16)
        lr_power = 0.9
        resume_training = False
        if model_name is 'AtrousFCN_Resnet50_16s':
            weight_decay = 0.0001 / 2
        else:
            weight_decay = 1e-4
    elif CONFIG == 'PASCAL':
        model_name = 'AtrousFCN_Resnet50_16s'
        # model_name = 'Atrous_DenseNet'
        # model_name = 'DenseNet_FCN'
        # batch_size = 16
        batch_size = 8
        batchnorm_momentum = 0.95
        epochs = 250
        lr_base = 0.01 * (float(batch_size) / 16)
        lr_power = 0.9
        resume_training = False
        if model_name is 'AtrousFCN_Resnet50_16s':
            weight_decay = 0.0001 / 2
        else:
            weight_decay = 1e-4
    elif CONFIG == 'COCO':
        # model_name = 'AtrousFCN_Resnet50_16s'
        #model_name = 'Atrous_DenseNet'
        model_name = 'DenseNet_FCN'
        # model_name = 'FCN_Resnet50_32s'
        batch_size = 2
        batchnorm_momentum = 0.95
        epochs = 450
        lr_base = 0.2 * (float(batch_size) / 4)
        lr_power = float(1)/float(30)
        resume_training=False
        weight_decay = 0.0001/2
        target_size = (320, 320)
    else:
        print("Config not set correctly, check the code.")
        exit(1)
    ##########################################
    if dataset == 'IIT-AFF':
        path_prefix = '/home/niu/Liang_Niu3/IIT_Affordances_2017/'
        train_file_path = os.path.join(path_prefix, 'fcn_test.txt')
        val_file_path   = os.path.join(path_prefix, 'fcn_val.txt')
        # data_dir        = os.path.join(path_prefix, 'rgb_origin')
        data_dir        = os.path.join(path_prefix, 'rgb') # 512x512 images
        target_size = (512, 512)
        label_dir       = os.path.join(path_prefix, 'affordances_labels_png')
        data_suffix='.jpg'
        label_suffix='.png'
        # 0-background,1-contain,2-cut,3-display,4-engine,5-grasp
        # 6-hit,7-pound,8-support,9-w-grasp (wrap grasp)
        classes = 10
        class_weight = None


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size,
          epochs,
          lr_base,
          lr_power,
          weight_decay,
          classes,
          model_name,
          train_file_path,
          val_file_path,
          data_dir,
          label_dir,
          target_size=target_size,
          batchnorm_momentum=batchnorm_momentum,
          resume_training=resume_training,
          class_weight=class_weight,
          dataset=dataset,
          data_suffix=data_suffix,
          label_suffix=label_suffix,
          log_dir='logs_aff',
          )
