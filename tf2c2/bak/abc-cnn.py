import numpy as np
import tensorflow as tf

# 加载之前转存处理好的数据
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
# One hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


# TensorFlow使用GPU时，在我的1050上，无法占用全部内存
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1300)])
tf.config.experimental.set_memory_growth = True

# 模型定义
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(n_classes, activation="softmax")
])

# 绘制神经网络图
model.summary()
# 开始训练
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
model.save("abc_model/1")
