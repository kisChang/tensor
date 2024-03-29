import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import abcutils

model = tf.keras.models.load_model('model_abc/1')


def outLayerImage():
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # TODO 从训练集中准备一个随机的输入图像
    # horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
    # human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
    # img_path = random.choice(horse_img_files + human_img_files)
    img_path = 'E:\\jupyter_tensor\\abc-data\\train\\0\\img060-006.png'

    img = load_img(img_path, color_mode="grayscale", target_size=(90, 90))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                # if x > 0:
                #   x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


def outGenImage():
    train_datagen, test_datagen, input_shape = abcutils.getAbcGen(genDataDir='C:\\Users\\KisChang\\Desktop\\abc-gen')
    for i in range(9):
        train_datagen.next()


(train_datagen, test_datagen, input_shape) = abcutils.getAbcGen()


label, count = np.unique(train_datagen.labels, return_counts=True)
print(label, count)
fig = plt.figure()
plt.bar(label, count, width=0.7, align='center')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0, 7500)
for a, b in zip(label, count):
    plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
plt.show()
