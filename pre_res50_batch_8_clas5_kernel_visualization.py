from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.models import Model
import matplotlib.pyplot as plt
import time



# GPU 显存按需分配调用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)


def get_img_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.
    return img_tensor

def get_layer(layer_name,img_path):
    layer_outputs = model.get_layer(layer_name).output
    # Creates a model that will return these outputs, given the model input:
    intermediate_layer_model = Model(inputs=model.input, outputs=layer_outputs)
    intermediate_output = intermediate_layer_model.predict(get_img_tensor(img_path))
    return intermediate_output

def get_layer_names(model):
    layer_names = []
    for layer in model.layers[1:174]:
        layer_names.append(layer.name)
    return layer_names

def get_kernel_vis(model,img_path):
    for layer_name in get_layer_names(model):
        layer_activation = get_layer(layer_name,img_path)
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        time.sleep(1)
        plt.savefig('B:\\pre_res_50_kernel_clas_5_batch_8\\'+layer_name+'.jpg')
        print(layer_name + '   save done!')


if __name__ == '__main__':
    images_per_row = 32
    #model
    model=load_model('B:\\my_model\\pre_res_50_clas_5_batch_8.h5')
    model.summary()  # As a reminder.
    img_path="B:\\vis_test_img.jpg"
    get_kernel_vis(model,img_path)