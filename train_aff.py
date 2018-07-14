## This is the training code for Liang Niu's experiments
## for affordance/segmentation co-segmentation network
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
    CONFIG = 'SEG_AFF' # 'SEG_AFF'->train_seg_aff(), 'DEBUG' -> testing TODO: more configurations for experiments

    ############ Configuration #################
    if CONFIG == 'SEG_AFF':
        model_name = 'AtrousFCN_Resnet50_Aff_Res'
        seg_classes = 80 # 80 + 1, coco
        aff_classes = 10 # 9 + 1, iitaff
        # 0-background,1-contain,2-cut,3-display,4-engine,5-grasp
        # 6-hit,7-pound,8-support,9-w-grasp (wrap grasp)
        batch_size = 8
        batchnorm_momentum = 0.95
        epochs = 250
        lr_base = 0.01 * (float(batch_size) / 16)
        lr_power = 0.9
        resume_training = False
        weight_decay = 0.0001 / 2
    elif CONFIG == 'DEBUG':
        model_name = 'AtrousFCN_Resnet50_Aff'
        weight_decay = 0.0001 / 2
        batchnorm_momentum = 0.95
        input_shape = (512,512,3)
        seg_classes = 21
        aff_classes = 10
        model = globals()[model_name](weight_decay=weight_decay,
                              input_shape=input_shape,
                              batch_momentum=batchnorm_momentum,
                              seg_classes=seg_classes,
                              aff_classes=aff_classes)
        model.summary()
        exit(0)
    else:
        print("Config not set correctly, check the code.")
        exit(1)
    ############ Datasets Setting ############
    if dataset == 'IIT-AFF':
        from config import DATASET_PATH
        #path_prefix = '/home/niu/Liang_Niu3/IIT_Affordances_2017/'
        path_prefix = DATASET_PATH
        train_file_path = os.path.join(path_prefix, 'fcn_train_and_val.txt')
        val_file_path   = os.path.join(path_prefix, 'fcn_test.txt')
        # data_dir        = os.path.join(path_prefix, 'rgb_origin')
        data_dir        = os.path.join(path_prefix, 'rgb') # 512x512 images
        target_size = (512, 512)
        aff_label_dir       = os.path.join(path_prefix, 'affordances_labels_png')
        semantic_label_dir = os.path.join(path_prefix, "semantic_labels_png")
        data_suffix='.jpg'
        label_suffix='.png'
        class_weight = None


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    if CONFIG == 'SEG_AFF':
        train_seg_aff(batch_size,
                      epochs,
                      lr_base,
                      lr_power,
                      weight_decay,
                      seg_classes,
                      aff_classes,
                      model_name,
                      train_file_path,
                      val_file_path,
                      data_dir,
                      aff_label_dir,
                      semantic_label_dir,
                      target_size=target_size,
                      batchnorm_momentum=batchnorm_momentum,
                      resume_training=resume_training,
                      class_weight=class_weight,
                      dataset=dataset,
                      data_suffix=data_suffix,
                      label_suffix=label_suffix,
                      log_dir='logs_segaff',
                      )
        print("Done Training.")
    else:
        print("Wrong Congiguration")
