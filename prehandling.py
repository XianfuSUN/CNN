from PIL import Image
import numpy as np
import random


# dataset of the satellite map from 4 categories
# 1 for residential, 2 for industrial, 3 for downtown, 4 for countryside
# encode as one-hot vector
class DataSet:
    def __init__(self):
        self.trainData = []
        self.testData = []
        self.curIndex = 0
        self.testData = []

    # read and parse image into 160 * 160
    # mode 1 for reading training data
    # mode 0 for reading testing data
    def read_img(self, path, label, mode=0):
        cur_img = Image.open(path).convert("L")
        for left in range(10):
            for top in range(6):
                box = (left * 160, top * 160, (left + 1) * 160, (top + 1) * 160)
                crop = cur_img.crop(box)
                data = np.asarray(crop) / 255
                truth = [0, 0, 0, 0]
                truth[label - 1] = 1
                if mode == 0:
                    self.trainData.append((data, np.array(truth)))
                elif mode == 1:
                    self.testData.append((data, np.array(truth)))
                # self.trainData.append((data_random, np.array([truth])))

    # crop the images from different categories
    # the size of each image is 160 * 160
    def init_trainData(self):
        print("start to read residential images....")
        for num in range(1, 11):
            cur_path = "data/Residential/r" + str(num) + ".jpg"
            self.read_img(cur_path, 1)

        print("start to read industrial images....")
        for num in range(1, 11):
            cur_path = "data/Industrial/i" + str(num) + ".jpg"
            self.read_img(cur_path, 2)

        print("start to read Downtown images....")
        for num in range(1, 11):
            cur_path = "data/Downtown/d" + str(num) + ".jpg"
            self.read_img(cur_path, 3)

        print("start to read Countryside's images....")
        for num in range(1, 11):
            cur_path = "data/country/c" + str(num) + ".jpg"
            self.read_img(cur_path, 4)
        random.shuffle(self.trainData)

    def init_testData(self):
        print("start to read test residential images....")
        for num in range(1, 4):
            cur_path = "test/residential/r" + str(num) + ".jpg"
            self.read_img(cur_path, 1, mode=1)

        print("start to read test industrial images....")
        for num in range(1, 4):
            cur_path = "data/industrial/i" + str(num) + ".jpg"
            self.read_img(cur_path, 2, mode=1)

        print("start to read test downtown images....")
        for num in range(1, 4):
            cur_path = "data/Downtown/d" + str(num) + ".jpg"
            self.read_img(cur_path, 3, mode=1)

        print("start to read test Countryside's images....")
        for num in range(1, 4):
            cur_path = "data/country/c" + str(num) + ".jpg"
            self.read_img(cur_path, 4, mode=1)
        random.shuffle(self.testData)

    # return the next batch of data
    def nextBatch(self, batch, mode=0):
        x = []
        y = []
        dataSet = self.trainData if mode == 0 else self.testData
        for i in range(self.curIndex, self.curIndex + batch):
            x.append(dataSet[i][0].flatten())
            y.append(dataSet[i][1])
        x = np.array(x)
        y = np.array(y)
        self.curIndex += batch
        return x, y


if __name__ == "__main__":
    ds = DataSet()
    ds.init_trainData()
