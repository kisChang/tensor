import glob
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 所有数据集合目录
TRAIN_DATA_DIR_ABC = "E:\\jupyter_tensor\\abc-data\\abc\\train\\"
TEST_DATA_DIR_ABC = "E:\\jupyter_tensor\\abc-data\\abc\\test\\"
TRAIN_DATA_DIR_01 = "E:\\jupyter_tensor\\abc-data\\01\\train\\"
TEST_DATA_DIR_01 = "E:\\jupyter_tensor\\abc-data\\01\\test\\"


def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure()
    for i in range(9):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(331 + i)
        sub_img.imshow(img)
    plt.show()
    return fig


def getAbcGen(genDataDir=None):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                 # rotation_range=15,
                                 # width_shift_range=0.15, height_shift_range=0.15,
                                 # shear_range=0.2
                                 )
    train_datagen = datagen.flow_from_directory(TRAIN_DATA_DIR_ABC,
                                                batch_size=10, shuffle=True,
                                                class_mode='categorical',
                                                color_mode='grayscale',
                                                save_to_dir=genDataDir, save_prefix='gen',
                                                classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                                target_size=(90, 90))
    test_datagen = datagen.flow_from_directory(TEST_DATA_DIR_ABC,
                                               batch_size=20, shuffle=True,
                                               class_mode='categorical',
                                               color_mode='grayscale',
                                               save_to_dir=genDataDir, save_prefix='gen',
                                               classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                               target_size=(90, 90))
    input_shape = (90, 90, 1)
    return train_datagen, test_datagen, input_shape


def get01Gen(genDataDir=None):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                 # rotation_range=20,
                                 # width_shift_range=0.15, height_shift_range=0.15,
                                 # shear_range=0.2
                                 )
    train_datagen = datagen.flow_from_directory(TRAIN_DATA_DIR_01,
                                                batch_size=10, shuffle=True,
                                                class_mode='categorical',
                                                color_mode='grayscale',
                                                save_to_dir=genDataDir, save_prefix='gen',
                                                classes=['0', '1'],
                                                target_size=(90, 90))
    test_datagen = datagen.flow_from_directory(TEST_DATA_DIR_01,
                                               batch_size=20, shuffle=True,
                                               class_mode='categorical',
                                               color_mode='grayscale',
                                               save_to_dir=genDataDir, save_prefix='gen',
                                               classes=['0', '1'],
                                               target_size=(90, 90))
    input_shape = (90, 90, 1)
    return train_datagen, test_datagen, input_shape


def initGpu():
    # -TensorFlow使用GPU时，在我的1050上，无法占用全部内存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1300)])
    tf.config.experimental.set_memory_growth(gpus[0], True)


# 模型定义
def defineModel(num_classes, input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")

        # 1-9929 - 13 - 20201224-173243
        # tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        # tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        # tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        # tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(640, activation="relu"),
        # tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.summary() # 绘制神经网络图
    return model


def startFit(model, train_datagen, test_datagen, callbacks):
    import datetime
    log_dir = "logs-fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 测试中 9980 但是准确率只有14
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # 测试到了 9966 准确 16
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    # 这个上不去了 准确17
    # 9746 5
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model.fit(
        train_datagen,
        steps_per_epoch=3000,
        epochs=5000, verbose=2,
        validation_data=test_datagen,
        validation_steps=1000,
        callbacks=[
            tensorboard_callback,
            callbacks])


class myCallback(tf.keras.callbacks.Callback):
    last_accuracy = 0
    target = 9000
    savePath = None

    def __init__(self, target=9980, savePath=None):
        super().__init__()
        self.savePath = savePath
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        getAcc = int(logs.get('accuracy') * 10000)
        if getAcc > self.last_accuracy:
            self.last_accuracy = getAcc
            if self.savePath is not None:
                modelSavePath = os.path.join(self.savePath, self.last_accuracy.__str__())
                self.model.save(modelSavePath)
        if getAcc > self.target:
            print("\nReached {}% accuracy so cancelling training!".format(self.target))
            self.model.stop_training = True
