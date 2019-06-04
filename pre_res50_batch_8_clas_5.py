# coding=utf-8

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers  #正则化防止过拟合resnet50深度太大熵容量过大
from keras.models import load_model
import keras
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt

# GPU 显存按需分配调用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
ktf.set_session(session)

#写一个LossHistory类，保存loss和acc
class LossHistory1(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('B:\\my_model\\pre_res_50_clas5_batch_8_1.jpg')
        plt.show()

class LossHistory2(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('B:\\my_model\\pre_res_50_clas5_batch_8_2.jpg')
        plt.show()



NB_CLASS=5
IM_WIDTH=224
IM_HEIGHT=224
train_root='B:\\art_set2\\train'
vaildation_root='B:\\art_set2\\vaildationdata'
test_root='B:\\art_set2\\test'
batch_size=8
EPOCH=50





# train data
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=True
)

# vaild data
vaild_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
vaild_generator = train_datagen.flow_from_directory(
    vaildation_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=True
)

# test data
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=True
)


# 构建不带分类器的预训练模型
base_model = ResNet50(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加两个全连接层
x = Dense(2048, activation='relu',kernel_regularizer=regularizers.l1_l2(0.01,0.01),
            activity_regularizer=regularizers.l1_l2(0.01,0.01))(x)
x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l1_l2(0.01,0.01),
            activity_regularizer=regularizers.l1_l2(0.01,0.01))(x)
# 添加一个分类器，假设我们有5个类
predictions = Dense(5, activation='softmax',kernel_regularizer=regularizers.l1_l2(0.01,0.01),
            activity_regularizer=regularizers.l1_l2(0.01,0.01))(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

'''
model.summary()
p=0
for i in model.layers:
    print(i.name)
    print(p)
    p=p+1


if os.path.exists('B:\\my_model\\pre_res_50_clas_5_batch_8.h5'):
    model=load_model('B:\\my_model\\pre_res_50_clas_5_batch_8.h5')
else:
    model = Model(inputs=base_model.input, outputs=predictions)
'''
# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 resnet50的卷积层

for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['acc',top_k_categorical_accuracy])



# 在新的数据集上训练几代
history1 = LossHistory1()#保存历史数据1

reducelron = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                         min_delta=0.0001, cooldown=0, min_lr=0)
 #when acc doesn't go up, reduce the learning rate to have more acc
checkpoint = ModelCheckpoint(filepath='B:\\my_model\\pre_res_50_clas_5_batch_8.h5', monitor='val_acc', verbose=1, save_best_only=True,mode='max')


#save best
model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size,
                            callbacks=[history1,checkpoint,reducelron],validation_steps=vaild_generator.n/batch_size)
# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。
history1.loss_plot('epoch')

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 我们选择训练最上面的142层
# 留最后一层卷积层
for layer in model.layers[:142]:
   layer.trainable = False
for layer in model.layers[142:]:
   layer.trainable = True


history2 = LossHistory2()#保存历史数据2
# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['acc',top_k_categorical_accuracy])

# 我们继续训练模型
# 训练最后一层卷积层
model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size,
                            callbacks=[history2,checkpoint,reducelron],validation_steps=vaild_generator.n/batch_size)
history2.loss_plot('epoch')
model=load_model('B:\\my_model\\pre_res_50_clas_5_batch_8.h5')

loss,acc,top_acc=model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
print ('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))

# structure of model!
from keras.utils.vis_utils import plot_model
model=load_model('B:\\my_model\\pre_res_50_clas_5_batch_8.h5')
plot_model(model, to_file='B:\\pre_res_50.png', show_shapes=True)
    