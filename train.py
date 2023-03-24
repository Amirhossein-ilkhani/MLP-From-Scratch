import datasets
import nets
import utils
import numpy as np
from sklearn.model_selection import train_test_split



def main():

    # -------------------------- loading cifar 10 -----------------------------
    x_train, x_test, y_train, y_test, label_name = datasets.load_cifar_10()
    y_train = np.array(y_train)
    y_train = y_train.reshape((50000,1))
    y_test = np.array(y_test)
    y_test = y_test.reshape((10000,1))
    print(x_train.shape)

    # -------------------------- plot cifar 10 -----------------------------
    datasets.cifar_10_plot(x_train, y_train, label_name)

    # --------------------------- read yaml file --------------------------------
    yaml = utils.load()
    print(yaml)

    # -------------------------- normalization/Standardization ----------------------------------
    if(yaml['input'] == 'Normalized'):
        X_train = (x_train-np.min(x_train))/(np.max(x_train) - np.min(x_train))
        X_test = (x_test-np.min(x_test))/(np.max(x_test) - np.min(x_test))
    elif(yaml['input'] == 'Standardaized'):
        X_train = (x_train - np.mean(x_train, axis=1).reshape(x_train.shape[0],1)) / (np.std(x_train, axis=1).reshape(x_train.shape[0],1))
        X_test = (x_test - np.mean(x_test, axis=1).reshape(x_test.shape[0],1)) / (np.std(x_test, axis=1).reshape(x_test.shape[0],1))
    else:
        X_train = x_train
        X_test = x_test

    print(X_train.shape)
    print(X_test.shape)
    print(X_train[0])
    #--------------------------- train - validation splitting -----------------
    X_train_1, X_valid, y_train_1, y_valid = train_test_split(X_train,y_train,test_size=0.1,stratify=y_train)
    print(X_train_1.shape)
    print(X_valid.shape)

    # -------------------------- one hot encoding ------------------------------
    Y_train_1 = np.zeros((y_train_1.shape[0], y_train_1.max()+1)) 
    Y_train_1[np.arange(y_train_1.size), y_train_1.T] = 1
    Y_test = np.zeros((y_test.shape[0], y_test.max()+1)) 
    Y_test[np.arange(y_test.size), y_test.T] = 1
    Y_valid = np.zeros((y_valid.shape[0], y_valid.max()+1)) 
    Y_valid[np.arange(y_valid.size), y_valid.T] = 1


    #-------------------------------- training ---------------------------------
    X_train_2 = X_train_1[0:1000, :]
    Y_train_2 = Y_train_1[0:1000, :]

    model = nets.MLP(inputs_layer=yaml['input_layer'], hidden_layer=yaml['hidden_layer'], output_layer=yaml['output_layer'], eta=yaml['learning_rate'], mu=yaml['mu'], sigma=yaml['sigma'], af=yaml['af'], loss_function=yaml['loss'], af_last_layer= yaml['af_last_layer'], momentum= yaml['momentum'])


    model.train(epoch=yaml['epochs'], batch=yaml['batch'], x_train=X_train_1, y_train=Y_train_1, x_valid=X_valid, y_valid=Y_valid, x_test=X_test, y_test=Y_test)



if __name__ == '__main__':main()