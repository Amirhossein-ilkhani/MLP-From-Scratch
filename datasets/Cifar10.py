import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

def extract(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f,encoding='latin1')
    return dict


def load():
    #----------------------------------load_batches--------------------------------
    file = r'datasets\data_batch_1'
    data_batch_1 = extract(file)
    file = r'datasets\data_batch_2'
    data_batch_2 = extract(file)
    file = r'datasets\data_batch_3'
    data_batch_3 = extract(file)
    file = r'datasets\data_batch_4'
    data_batch_4 = extract(file)
    file = r'datasets\data_batch_5'
    data_batch_5 = extract(file)
    file = r'datasets\test_batch'
    test_batch = extract(file)
    
    # print(type(data_batch_1))
    # print(data_batch_1.keys())
    # for item in data_batch_1:
    #     print(item, type(data_batch_1[item]))
    # print("Labels:", set(data_batch_1['labels']))
    # X_train = data_batch_1['data']
    # Y_train = data_batch_1['labels']
    # print(X_train.shape)
    # print(len(Y_train))

    # -------------------------------make train data and test data --------------------------------
    X_train = data_batch_1['data']
    X_train = np.append(X_train, data_batch_2['data'], axis=0)
    X_train = np.append(X_train, data_batch_3['data'], axis=0)
    X_train = np.append(X_train, data_batch_4['data'], axis=0)
    X_train = np.append(X_train, data_batch_5['data'], axis=0)
    X_test = test_batch['data']
    # print(X_train.shape)
    # print(X_test.shape)
    Y_train = data_batch_1['labels']
    Y_train = Y_train + data_batch_2['labels'] + data_batch_3['labels'] + data_batch_4['labels'] + data_batch_5['labels']
    Y_test = test_batch['labels']
    # print(len(Y_train))
    # print(len(Y_test))

    # ----------------------------------------label_names --------------------------------
    file = r'datasets\batches.meta'
    meta = extract(file)
    # print(type(meta))
    # print(meta.keys())
    label_name = meta['label_names']
    # print("Label Names:", label_name)

    # ----------------------------------------reshape data --------------------------------
    # X_train = X_train.reshape(len(X_train),3,32,32)
    # X_train = X_train.transpose(0,2,3,1)
    # X_test = X_test.reshape(len(X_test),3,32,32)
    # X_test = X_test.transpose(0,2,3,1)
    # print(X_train.shape)
    # print(X_test.shape)

    return X_train, X_test, Y_train, Y_test, label_name




def visualize(x_train, y_train, label_name):
    X_train = x_train.reshape(len(x_train),3,32,32)
    X_train = X_train.transpose(0,2,3,1)

    list_split = [[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(y_train)):
        if y_train[i] == 0:
            list_split[0].append(i)
        elif y_train[i] == 1:
            list_split[1].append(i)
        elif y_train[i] == 2:
            list_split[2].append(i)
        elif y_train[i] == 3:
            list_split[3].append(i)
        elif y_train[i] == 4:
            list_split[4].append(i)
        elif y_train[i] == 5:
            list_split[5].append(i)
        elif y_train[i] == 6:
            list_split[6].append(i)
        elif y_train[i] == 7:
            list_split[7].append(i)
        elif y_train[i] == 8:
            list_split[8].append(i)
        elif y_train[i] == 9:
            list_split[9].append(i)

    plt.figure()
    columns = 10
    rows = 10
    for r in range(10):
        for c in range(10):
            plt.subplot(10,10,r*10+c+1)
            img =  X_train[list_split[r][random.randrange(0, 5000, 10)]]
            plt.imshow(img)
    
    plt.show()


def main():
    
    x_train, x_test, y_train, y_test, label_name = load()
    visualize(x_train, y_train, label_name)

if __name__ == "__main__":main()
