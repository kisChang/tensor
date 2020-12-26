import tensorflow as tf
import abcutils

abcutils.initGpu()
(train_datagen, test_datagen, input_shape) = abcutils.getAbcGen()
# 模型加载
model = tf.keras.models.load_model('model_abc/1')
# 开始训练
callbacks = abcutils.myCallback(savePath="model_abc", target=9990)
history = abcutils.startFit(model, train_datagen, test_datagen, callbacks, log_dir_arg='logs-fit/20201225-163838')
model.save("model_abc/1")