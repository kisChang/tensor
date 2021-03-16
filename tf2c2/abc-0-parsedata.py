import glob
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = "E:\\jupyter_tensor\\tf2c2\\"
TRAIN_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\abc\\train\\"
TEST_DATA_DIR = "E:\\jupyter_tensor\\abc-data\\abc\\test\\"
dirMap = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
}
dirMap2 = {
    '41': 0,     # 'A',
    '42': 1,     # 'B',
    '43': 2,     # 'C',
    '44': 3,     # 'D',
    '45': 4,     # 'E',
    '46': 5,     # 'F',
    '47': 6,     # 'G',
    '48': 7,     # 'H',
    '49': 8,     # 'I',
    '4a': 9,     # 'J',
}


def loadImg(path, label, image_list, label_list):
    for filename in glob.glob(path + '\\**\\*.png', recursive=True):
        img = Image.open(filename)
        img = img.resize((90, 90), Image.ANTIALIAS)  # 等比例缩放
        img = img.convert('L')  # 转灰度
        img = ImageOps.invert(img)  # 反转，白底黑字
        image_list.append(np.array(img))
        label_list.append(label)
    return image_list, label_list


def loadTrain():
    train_images = []
    train_labels = []
    for name in dirMap:
        print('parse >>' + name)
        train_images, train_labels = loadImg(TRAIN_DATA_DIR + name, dirMap[name], train_images, train_labels)
        print('parse >>' + name + "  ok")
    train_images = np.array(train_images, dtype=np.float32)
    print(train_images.shape)
    np.savez_compressed(BASE_DIR + "bin-data\\train_images", train_images)
    np.save(BASE_DIR + "bin-data\\train_labels", train_labels)


def loadTest():
    test_images = []
    test_labels = []
    for name in dirMap:
        print('parse >>' + name)
        test_images, test_labels = loadImg(TEST_DATA_DIR + name, dirMap[name], test_images, test_labels)
        print('parse >>' + name + "  ok")
    test_images = np.array(test_images, dtype=np.float32)
    print(test_images.shape)
    np.savez_compressed(BASE_DIR + "bin-data\\test_images", test_images)
    np.save(BASE_DIR + "bin-data\\test_labels", test_labels)


loadTrain()
loadTest()
print("save done.")

