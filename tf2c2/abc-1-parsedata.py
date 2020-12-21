import tensorflow as tf
import glob
import numpy as np
from PIL import Image, ImageOps

# 这个数据集 500张图片，作为测试集
TRAIN_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\chart7k\\train\\"

def loadImg(path):
    image = []
    text = []
    for filename in glob.glob(path + '**\\*.png', recursive=True):
        img = Image.open(filename)
        img = img.resize((int(img.size[0] * 0.1), int(img.size[1] * 0.1)), Image.ANTIALIAS)  # 等比例 缩放0.1
        img = img.crop((15, 0, 105, 90))  # (left, upper, right, lower) 裁剪成90 * 90
        img = img.convert('L')  # 转灰度
        img = ImageOps.invert(img)  # 反转，白底黑字
        image.append(np.array(img))
        filename = filename.lstrip(path)
        label = filename[filename.index('mg0') + 3:].rstrip('.png')
        label = label[0:2]
        text.append(int(label) - 11)
    return image, text


test_images, test_labels = loadImg(TRAIN_DATA_DIR)
# 转 np
test_images = np.array(test_images, dtype=np.float32)
print(test_images.shape)
np.save("bin-data/test_images.npy", test_images)
np.save("bin-data/test_labels.npy", test_labels)

print("save done.")