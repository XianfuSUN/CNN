import tensorflow.compat.v1 as tf
import numpy as np
from prehandling import DataSet
import random
import sys
from matplotlib import pyplot as plt

# disable tensorflow v2
tf.disable_v2_behavior()


# CNN class:
# initial the variables for a CNN network
# provide methods: convolve, fully connected and train
class CNN:
    def __init__(self, width, height, output_class, dataset):
        self.width = width
        self.height = height
        self.output_class = output_class
        self.ds = dataset
        # parameters for convolve layers
        self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=1e-1))  # kernel 1: 5*5*16
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        self.W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=1e-1))  # kernel 2: 5*5*32
        self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        self.W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=1e-1))  # kernel 3: 3*3*64
        self.b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
        self.W_conv4 = tf.Variable(tf.truncated_normal([5, 5, 128, 128], stddev=1e-1))  # kernel 4: 3*3*64
        self.b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
        # parameters for fully connected layers
        self.W_fc1 = tf.Variable(tf.truncated_normal([12800, 3200], stddev=1e-2))  # hidden layer1: 6400->1600
        self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[3200]))
        self.W_fc2 = tf.Variable(tf.truncated_normal([3200, 1600], stddev=1e-2))  # hidden layer2:1600->800
        self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[1600]))
        # parameters for output layer
        self.W_fc3 = tf.Variable(tf.truncated_normal([1600, 4], stddev=1e-2))  # output layer: 800->4
        self.b_fc3 = tf.Variable(tf.constant(0.1, shape=[4]))

    # convolution layer, using the same padding strategy
    # include relu activation and max_pooling
    # return the tensor after max_pooling
    def conv_layer(self, data, W, bias):
        convolve = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding="SAME") + bias
        activate = tf.nn.relu(convolve)
        layer = tf.nn.max_pool(activate, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return layer

    # fully connected layer
    # return the tensor after relu activation
    def fc_layer(self, data, W, bias):
        fc = tf.matmul(data, W) + bias
        activate = tf.nn.relu(fc)
        return activate

    # output layer
    def output_layer(self, data, W, bias):
        fc = tf.matmul(data, W) + bias
        activate = tf.nn.softmax(fc)
        return activate

    # feed forward the network
    def forward(self, x_input):
        # convolve layers
        layer1 = self.conv_layer(x_input, self.W_conv1, self.b_conv1)
        # the output should be 80*80*16
        layer2 = self.conv_layer(layer1, self.W_conv2, self.b_conv2)
        # the output should be 40*40*32
        layer3 = self.conv_layer(layer2, self.W_conv3, self.b_conv3)
        # the output should be 20*20*64
        layer4 = self.conv_layer(layer3, self.W_conv4, self.b_conv4)
        # the output should be 10*10*64
        fc_dim = 10 * 10 * 128
        layer = tf.reshape(layer4, [-1, fc_dim])

        # fully connected layers
        fc_1 = self.fc_layer(layer, self.W_fc1, self.b_fc1)
        fc_2 = self.fc_layer(fc_1, self.W_fc2, self.b_fc2)
        # output layer
        y_CNN = self.output_layer(fc_2, self.W_fc3, self.b_fc3)
        return y_CNN, layer1, layer2, layer3

    # train the CNN network
    def train(self, data, mode=1):
        # placeholders for data input
        x = tf.placeholder(tf.float32, shape=[None, self.width * self.height])
        y_label = tf.placeholder(tf.float32, shape=[None, self.output_class])
        x_input = tf.reshape(x, [-1, self.width, self.height, 1])

        sess = tf.InteractiveSession()
        y_CNN, layer1, layer2, layer3 = self.forward(x_input)

        # set up loss function
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-5, global_step, 500, 0.98)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_CNN), reduction_indices=[1]))
        train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        # run the session
        print("begin to run")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        iterations = 240  # max iterations:240, batch size: 10
        training_epochs = 11
        Loss = []
        trained_times = 0
        med_images = []
        # train_mode
        if mode == 1:
            Accuracy = []
            for epoch in range(training_epochs):
                print("______epoch %s_____" % epoch)
                self.ds.curIndex = 0
                random.shuffle(self.ds.trainData)  # shuffle the dataset
                for i in range(iterations):
                    batch = self.ds.nextBatch(10)
                    trained_times += 1
                    train.run(feed_dict={x: batch[0], y_label: batch[1]})
                    if i % 10 == 0:
                        loss, accuracy_temp = sess.run([cross_entropy, accuracy], feed_dict={x: batch[0], y_label: batch[1]})
                        Loss.append(loss)
                        Accuracy.append(accuracy_temp)
                        print("loss: ", loss)
                        print("accuracy", accuracy_temp)

                    # write some of the intermediate output
                    if trained_times == 1 or trained_times == 1000 or trained_times == 5000\
                            or trained_times == 100000:
                        med_result = sess.run([layer1, layer2, layer3], feed_dict={x: batch[0], y_label: batch[1]})
                        layer1_img = med_result[0][0]
                        layer1_img = np.reshape(layer1_img, [32, 80, 80])
                        img_mix = np.zeros([80, 80])
                        for c in range(32):
                            img_mix += layer1_img[c]
                        img_mix = img_mix * 255 / 32
                        print(img_mix)
                        med_images.append(img_mix)

            print("training finished")
            print("begin to calculate train error")
            data.curIndex = 0
            total_accurate = 0

            for j in range(24):
                batch = data.nextBatch(100)
                accuracy_rate = accuracy.eval(feed_dict={x: batch[0], y_label: batch[1]})
                total_accurate += accuracy_rate * 100
                print("accurate number: ", accuracy_rate * 100)
            print("train_accuracy:", total_accurate / 2400)
            # save the model
            saver.save(sess, "model/cnn")
            sess.close()
            return Loss, Accuracy, med_images
        # predict mode
        elif mode == 0:
            new_saver = tf.train.import_meta_graph("./model/cnn.meta")
            new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
            print("model restored")

            data.curIndex = 0
            predict_accuracy = []
            total_accurate = 0
            for i in range(9):
                batch = data.nextBatch(80, mode=1)
                accuracy_rate = accuracy.eval(feed_dict={x: batch[0], y_label: batch[1]})
                total_accurate += accuracy_rate * 80
                acc_accuracy = total_accurate / ((i + 1) * 80)
                predict_accuracy.append(acc_accuracy)
                print("accurate number: ", accuracy_rate * 80)
            print("test_accuracy:", total_accurate / 720)
            return predict_accuracy


# write the output
def write_output(path, data):
    with open(path, 'wt') as fd:
        for i in range(len(data)):
            fd.write(str(data[i]))
            fd.write("\n")
        fd.close()


def writeMatrix(path, data):
    with open(path, 'wt') as fd:
        for row in range(len(data)):
            for col in range(len(data[row])):
                fd.write(str(data[row][col]))
                if not col == len(data[row]) - 1:
                    fd.write(",")
            fd.write("\n")
        fd.close()


if __name__ == "__main__":
    ds = DataSet()
    ds.init_trainData()
    ds.init_testData()
    cnn = CNN(160, 160, 4, ds)
    if sys.argv[1] == "train":
        loss, acc_accuracy, med_img = cnn.train(ds)
        try:
            write_output("output/loss3.csv", loss)
            write_output("output/accumulative accuracy3.csv", acc_accuracy)
            for i in range(len(med_img)):
                path = "output/medium_img" + str(i) + ".csv"
                writeMatrix(path, med_img[i])
        except IOError:
            print("write output failed")

        # testing on data
    else:
        acc = cnn.train(ds, mode=0)
        print(acc)








