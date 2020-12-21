import numpy as np
import tensorflow as tf

TRAIN_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\chart7k\\train\\"
TEST_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\chart7k\\test\\"


train_images = np.load("bin-data\\train_images.npy")
train_labels = np.load("bin-data\\train_labels.npy")
test_images = np.load("bin-data\\test_images.npy")
test_labels = np.load("bin-data\\test_labels.npy")
print(train_images.shape)
print(test_images.shape)

# 数据规范化
n_classes = 10
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 90, 90, 1)
test_images = test_images / 255.0
test_images = test_images.reshape(test_images.shape[0], 90, 90, 1)
input_shape = (90, 90, 1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
tf.config.experimental.set_memory_growth = True


model = tf.keras.models.load_model('abc_model/1')
model.summary()
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
model.save("abc_model/2")
