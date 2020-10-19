# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_train(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        # gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

# '/home/dataset/GTSRB/Testing/00000/GT-00000.csv'
def readTrafficSigns_test(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    # for c in range(0,43):
    #     prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
    #     gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
    #     gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    #     # gtReader.next() # skip header
    #     next(gtReader)
    #     # loop over all images in current annotations file
    #     for row in gtReader:
    #         images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
    #         labels.append(row[7]) # the 8th column is the label
    #     gtFile.close()
    # return images, labels
    
    gtFile = open(rootpath + '/' + 'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader) 
    for row in gtReader:
        images.append(plt.imread(rootpath + '/' + row[0]))
        labels.append(row[7])
    return images, labels


if __name__ == "__main__":
    train_images, train_labels = readTrafficSigns_train("/home/dataset/GTSRB/Training")
    print(len(train_images))
    print(len(train_labels))
    test_images, test_labels = readTrafficSigns_test("/home/dataset/GTSRB/Testing")
    print(len(test_images))
    print(len(test_labels))
    print(test_labels)