
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from sklearn.metrics import confusion_matrix 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import itertools
import time

NB_CLASS=5
IM_WIDTH=224
IM_HEIGHT=224
train_root='B:\\art_set2\\train'
vaildation_root='B:\\art_set2\\vaildationdata'
test_root='B:\\art_set2\\test'
batch_size=1
EPOCH=30



vaild_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
vaild_generator = vaild_datagen.flow_from_directory(
    vaildation_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=False
)

model_path_res = 'B:\\my_model\\re_set2_batch_8_res50.h5'
model_path_vgg = 'B:\\my_model\\re_set2_batch_8_vg19.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
ktf.set_session(session)

model = load_model(model_path_vgg)


# 混淆矩阵
def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix of vgg19',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    time.sleep(1)
    plt.savefig('B:\\confusion_matrix_vgg.jpg')



pred_y = model.predict_generator(vaild_generator,steps=vaild_generator.n / batch_size)
predict_label = np.argmax(pred_y, axis=1)
true_label = vaild_generator.classes



'''
import pandas as pd
pd.crosstab(true_label,predict_label,rownames=['label'],colnames=['predict'])
'''

confusion_mat = confusion_matrix(true_label, predict_label)

plot_sonfusion_matrix(confusion_mat, classes = range(5))

