import tensorflow as tf
import abcutils

abcutils.initGpu()
(train_datagen, test_datagen, input_shape) = abcutils.getAbcGen()
# 模型加载
model = tf.keras.models.load_model('model_01/1')
# 开始训练
callbacks = abcutils.myCallback(savePath="model_01")
history = abcutils.startFit(model, train_datagen, test_datagen, callbacks)
model.save("model_01/1")