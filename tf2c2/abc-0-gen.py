import abcutils

abcutils.initGpu()

(train_datagen, test_datagen, input_shape) = abcutils.getAbcGen()
# 采用模型定义
model = abcutils.defineModel(train_datagen.num_classes, input_shape)
# 开始训练
callbacks = abcutils.myCallback(savePath="model_abc")
history = abcutils.startFit(model, train_datagen, test_datagen, callbacks)
model.save("model_abc/1")
