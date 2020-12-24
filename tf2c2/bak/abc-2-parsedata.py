import glob
import numpy as np
from PIL import Image, ImageOps

# 合并了这两个数据集作为训练集
TRAIN_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\nist\\by_merge\\"
TEST_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\chart7k\\test\\"


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


dirMap = {
    '41': 0,     # 'A',
    '42': 1,     # 'B',
    '43_63': 2,  # 'C',
    '44': 3,     # 'D',
    '45': 4,     # 'E',
    '46': 5,     # 'F',
    '47': 6,     # 'G',
    '48': 7,     # 'H',
    '49': 8,     # 'I',
    '4a_6a': 9,  # 'J',
}


def loadImgByNist(path, label, image_list, label_list):
    for filename in glob.glob(path + '\\**\\*.png', recursive=True):
        img = Image.open(filename)
        img = img.resize((90, 90), Image.ANTIALIAS)  # 等比例缩放
        img = img.convert('L')  # 转灰度
        img = ImageOps.invert(img)  # 反转，白底黑字
        image_list.append(np.array(img))
        label_list.append(label)
    return image_list, label_list


# 先加载一批另一个数据集的全部
train_images, train_labels = loadImg(TEST_DATA_DIR)
# 再加载第二个数据集
for name in dirMap:
    print('parse >>' + name)
    train_images, train_labels = loadImgByNist(TRAIN_DATA_DIR + name, dirMap[name], train_images, train_labels)
    print('parse >>' + name + "  ok")
train_images = np.array(train_images, dtype=np.float32)
print(train_images.shape)

np.save("bin-data/train_images.npy", train_images)
np.save("bin-data/train_labels.npy", train_labels)

print("save done.")