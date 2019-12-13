from matplotlib import pyplot as plt
import copy

path1 = "output/medium_img0.csv"
path2 = "output/medium_img1.csv"


def read_file(path):
    image = []
    with open(path, 'rt') as fd:
        line = fd.readline()
        print(line)
        while line:
            line_split = line.split(",")
            row = []
            for i in range(len(line_split)):
                row.append(float(line_split[i]))
            print(row)
            image.append(copy.copy(row))
            line = fd.readline()
    return image


if __name__ == "__main__":
    image = read_file(path1)
    print(image)
    plt.imshow(image, cmap="gray")
    plt.show()
    image = read_file(path2)
    print(image)
    plt.imshow(image, cmap="gray")
    plt.show()
