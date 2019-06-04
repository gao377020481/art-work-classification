from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
ktf.set_session(session)


number = ['1','2','3','4','5','6']


model_path_res = 'B:\\my_model\\re_set2_batch_8_res50.h5'
model_path_vgg = 'B:\\my_model\\re_set2_batch_8_vg19.h5'


def decode(result):
    if result[0][0] > 0.5:
        name = 'Edgar_Degas'
    if result[0][1] > 0.5:
        name = 'Mikhail_Vrubel'
    if result[0][2] > 0.5:
        name = 'Pablo_Picasso'
    if result[0][3] > 0.5:
        name = 'Rembrandt'
    if result[0][4] >0.5:
        name = 'Vincent_van_Gogh'
    return name
    

def predict(img_path,model_path):
    img = load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = x/255
    img_input = np.expand_dims(x,axis=0)
    print(img_input.shape)

    model = load_model(model_path)

    result = model.predict(img_input)

    print(result)

    result = decode(result)

    print(result)

if __name__ == '__main__':
    #for i in number:
        #img_path = 'B:\\Edgar_Degas_'+i+'.jpg'
        img_path = 'B:\\Edgar_Degas_22.jpg'
        predict(img_path,model_path_res)

