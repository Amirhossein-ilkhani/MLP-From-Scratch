import numpy as np
import matplotlib.pyplot as plt
import pickle
import losses as ls

class MLP:

    def __init__(self, inputs_layer = [3], hidden_layer = [16], output_layer = [2], mu = 0,sigma = 1, af = 'Relu', eta = 0.001 , loss_function = 'Cross_Entropy', af_last_layer = 'softmax', momentum = 'False'):

        self.inputs_layer = inputs_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.mu = mu
        self.sigma = sigma
        self.af = af
        self.eta = eta
        self.loss_function = loss_function
        self.af_last_layer = af_last_layer
        self.momentum = momentum

        self.train_acc = np.zeros(1)
        self.train_acc_mean = np.zeros(1)
        self.L = np.zeros(1)
        self.L_mean = np.zeros(1)
    
        layers = self.inputs_layer + self.hidden_layer + self.output_layer
        print("shape of layers is = {}".format(layers)) 

        weights = []
        der = []
        Vderw = []
        for i in range(len(layers)-1):
            w = self.sigma * np.random.rand(layers[i], layers[i+1]) + self.mu
            d = np.zeros((layers[i], layers[i+1]))
            vd = np.zeros((layers[i], layers[i+1]))
            weights.append(w)
            der.append(d)
            Vderw.append(vd)
        
        # for i in range(len(layers)-1):
        #     w = np.zeros((layers[i], layers[i+1]))
        #     d = np.zeros((layers[i], layers[i+1]))
        #     weights.append(w)
        #     der.append(d)

        self.weights = weights
        self.der = der
        self.Vderw = Vderw

        self.bias = []
        self.b_der = []
        self.Vderb = []
        for i in range(1,len(layers)):
            b = np.zeros((layers[i]))
            b_d = np.zeros((layers[i]))
            vderb = np.zeros((layers[i]))
            self.bias.append(b)
            self.b_der.append(b_d)
            self.Vderb.append(vderb)


        self.output = []
        for i in range(len(layers)):
            o = np.zeros(layers[i])
            self.output.append(o)

        print("len of weight is = {}".format(len(self.weights)))
        print("len of der is = {}".format(len(self.der)))
        print("len of bias is = {}".format(len(self.bias)))
        print("len of b_der = {}".format(len(self.b_der)))
        print("len of output is = {}".format(len(self.output)))
        print("weight = {}".format(self.weights))
        print("der = {}".format(self.der))
        print("bias = {}".format(self.bias))
        print("b_der = {}".format(self.b_der))
        print("output = {}".format(self.output))



    def forward(self, input):
        i = 0
        self.output[i] = input
        
        for w,b in zip(self.weights, self.bias):
            wx = np.dot(self.output[i], w)
            wxb = np.add(wx , b)
            i+=1

            if(i==len(self.weights)):
                if(self.af_last_layer == 'softmax'):
                    self.output[i] = self.softmax(wxb)
                elif(self.af_last_layer == 'Relu'):
                    self.output[i] = self.ReLU(wxb)
                elif(self.af_last_layer == 'sigmoid'):
                    self.output[i] = self.sigmoid(wxb)
                elif(self.af_last_layer == 'Tanh'):
                    self.output[i] = self.tanh(wxb)
                elif(self.af_last_layer == 'Leaky_Relu'):
                    self.output[i] = self.LealyReLU(wxb)
                else:
                    self.output[i] = wxb
                    
                    
            elif(self.af == 'Relu'):
                self.output[i] = self.ReLU(wxb)
            elif(self.af == 'sigmoid'):
                self.output[i] = self.sigmoid(wxb)
            elif(self.af == 'Tanh'):
                self.output[i] = self.tanh(wxb)
            elif(self.af == 'Leaky_Relu'):
                self.output[i] = self.LealyReLU(wxb)
            else:
                self.output[i] = wxb

        # print("Output = {}".format(self.output[i]))
        return self.output[i]

        
    def backward(self, y_pred, y_true) :


        e = np.subtract(y_pred, y_true)
        



        m = y_pred.shape[0]
        for i in reversed(range(len(self.der))):
            if(i+1 == len(self.der)):
                z = e
            elif(self.af == 'Relu') :
                z = e * self.ReLU_d(self.output[i+1])
            elif(self.af == 'sigmoid') :
                z = e * self.sigmoid_d(self.output[i+1])
            elif(self.af == 'Tanh') :
                z = e * self.tanh_d(self.output[i+1])
            elif(self.af == 'Leaky_Relu'):
                z = e * self.LealyReLU_d(self.output[i+1])
            else:
                z = e

            # print("e[{}] = {}\nz = {}".format(i+1,e,z))
            # print("e[{}].shape = {}\nz.shape = {}".format(i+1, e.shape, z.shape))

            self.der[i] = (1/m) * np.dot(self.output[i].T, z) + (0.1 * self.weights[i])/m
            # + (0.9 * self.weights[i])/m
            self.b_der[i] = (1/m) * np.sum(z, axis=0, keepdims=True) 
            # print("der[{}] = {}".format(i,self.der[i]))
            # print("der[{}].shape = {}".format(i,self.der[i].shape))

            e = np.dot(z, self.weights[i].T)

    

    def GD(self):
        if(self.momentum == 'False'):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - (self.der[i] * self.eta)
                self.bias[i] = self.bias[i] - (self.b_der[i] * self.eta)
        else:
            for i in range(len(self.weights)):
                self.Vderw[i] = (0.9 * self.Vderw[i]) + (0.1 * self.der[i])
                self.weights[i] = self.weights[i] - (self.Vderw[i] * self.eta)

                self.Vderb[i] = (0.9 * self.Vderb[i]) + (0.1 * self.b_der[i])
                self.bias[i] = self.bias[i] - (self.Vderb[i] * self.eta)




    def train(self, epoch = 1, batch = 2, x_train = np.zeros((1)), y_train = np.zeros((1)), x_valid = np.zeros((1)), y_valid = np.zeros((1)),  x_test = np.zeros((1)), y_test = np.zeros((1))):
        print("####################### training started ############################")
        iterations = int(x_train.shape[0]/batch)
        loss_valid_mean = np.zeros(1) 
        y_hat_valid_mean = np.zeros(1) 
        loss_train_mean = np.zeros(1) 
        y_hat_train_mean = np.zeros(1)
        loss_test_mean = np.zeros(1) 
        y_hat_test_mean = np.zeros(1)      

        for i in range(epoch):
            for k in range(iterations):
                input = x_train[k*batch : (k+1)*batch, : ]
                y_true = y_train[k*batch : (k+1)*batch, : ]
                out = self.forward(input)
                # if(self.loss_function == 'Cross-Entropy'):
                #     loss = self.cross_entropy(y_true , out)
                # else:
                #     loss = self.MSE(y_true , out)

                # self.train_acc = np.append(self.train_acc , self.acc(self.decode(out),y_true))
                # self.L = np.append(self.L, loss)

                # print("y = {};\n y_pred = {};\n;".format(y_true ,out))
                # print("y.shape = {}; y_pred.shape = {};  loss.shape = {};".format(y_true.shape , out.shape, loss.shape))

                self.backward(out, y_true)
                self.GD()

                # print("weights After GD = {};".format(self.weights))
                # print("acc = {}\t\t loss={} ".format(np.mean(self.train_acc), np.mean(self.L)))
                # print("***********end of iterration{} of epoch {}***************".format(k, i))
            

            # self.train_acc_mean = np.append(self.train_acc_mean ,np.mean(self.train_acc))
            # self.L_mean = np.append(self.L_mean, np.mean(self.L))

            
            # print("acc = {}\t\t loss={} ".format(np.mean(self.train_acc), np.mean(self.L)))
            # print("\n-----------------------------------end of epoch{} --------------------------------------".format(i))

            y_hat_train = self.forward(x_train)
            y_hat_train_mean = np.append(y_hat_train_mean, self.acc(y_train, self.decode(y_hat_train)))
            if(self.loss_function == 'Cross-Entropy'):
                loss_train = ls.cross_entropy(y_train , y_hat_train)
            else:
                loss_train = ls.MSE(y_train , y_hat_train)
            loss_train_mean = np.append(loss_train_mean, np.mean(loss_train))


            y_hat_valid = self.forward(x_valid)
            y_hat_valid_mean = np.append(y_hat_valid_mean, self.acc(y_valid, self.decode(y_hat_valid)))
            if(self.loss_function == 'Cross-Entropy'):
                loss_valid = ls.cross_entropy(y_valid , y_hat_valid)
            else:
                loss_valid = ls.MSE(y_valid , y_hat_valid)
            loss_valid_mean = np.append(loss_valid_mean, np.mean(loss_valid))


            y_hat_test = self.forward(x_test)
            y_hat_test_mean = np.append(y_hat_test_mean, self.acc(y_test, self.decode(y_hat_test)))
            if(self.loss_function == 'Cross-Entropy'):
                loss_test = ls.cross_entropy(y_test , y_hat_test)
            else:
                loss_test = ls.MSE(y_test , y_hat_test)
            loss_test_mean = np.append(loss_test_mean, np.mean(loss_test))

            print("acc = {}\t\t loss={} ".format((self.acc(y_train, self.decode(y_hat_train))), (np.mean(loss_train))))
            print("acc_test = {}\t\t loss_test={} ".format((self.acc(y_test, self.decode(y_hat_test))), (np.mean(loss_test))))
            print("\n-----------------------------------end of epoch{} --------------------------------------".format(i))
            
            # if((self.acc(y_train, self.decode(y_hat_train)) - self.acc(y_test, self.decode(y_hat_test))) > 0.07):
            #     break

        print("####################### training finished ############################")

        with open('weight.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle)

        with open('bias.pickle', 'wb') as handle:
            pickle.dump(self.bias, handle)


        plt.figure()
        plt.subplot(121)
        plt.plot(y_hat_train_mean, color='red')
        plt.plot(y_hat_valid_mean, color='orange')
        plt.plot(y_hat_test_mean, color='green')
        plt.xticks(range(0,len(y_hat_train_mean)+1, 1))
        plt.xlabel('epoch')
        plt.ylabel('acc')

        plt.subplot(122)
        plt.plot(loss_train_mean, color='red')
        plt.plot(loss_valid_mean, color='orange')
        plt.plot(loss_test_mean, color='green')
        plt.xticks(range(0,len(loss_train_mean)+1, 1))
        plt.xlabel('epoch')
        plt.ylabel('error')

        plt.legend(["train", "valid", 'test'], loc ="lower right")
        plt.show()

    def predict(self,x, y):
        self.train_acc = np.zeros(1)
        self.train_acc_mean = np.zeros(1)
        self.L = np.zeros(1)
        self.L_mean = np.zeros(1)
        y_hat = np.zeros((y.shape[0], y.shape[1]))
        print(x.shape, y.shape)

        for i in range(x.shape[0]):
            input = x[i:i+1, :]
            out1 = self.forward(input)

            y1 = y[i].reshape((1,10))
            if(self.loss_function == 'Cross_Entropy'):
                loss = self.cross_entropy(y1, out1)
            else :
                loss = self.MSE(y1, out1)
            self.train_acc = np.append(self.train_acc , self.acc(self.decode(out1),y1))
            self.train_acc_mean = np.append(self.train_acc_mean ,np.mean(self.train_acc))
            self.L = np.append(self.L, loss)
            self.L_mean = np.append(self.L_mean, np.mean(self.L))

            y_hat[i] = out1
        
        plt.figure()
        plt.subplot(121)
        plt.plot(self.train_acc_mean, color='red')
        plt.xticks(range(0,len(self.train_acc_mean)+1, 100))
        plt.xlabel('data')
        plt.ylabel('acc')

        plt.subplot(122)
        plt.plot(self.L_mean, color='blue')
        plt.xticks(range(0,len(self.L_mean)+1, 100))
        plt.xlabel('data')
        plt.ylabel('error')

        plt.show()

        y_hat_1 = self.decode(y_hat) 
        return y_hat_1


    def decode(self, y):
        y_hat = np.zeros_like(y) 
        y_hat[np.arange(len(y)), y.argmax(1)] = 1
        return y_hat


    def acc(self, y1, y2):
        sum = 0
        for i in range(y1.shape[0]):
            if(y1[i] == y2[i]).all():
                sum+=1
        
        return sum/y1.shape[0]
    
    def softmax(self, x):
        tmp = np.exp(x /50)
        s = np.sum(tmp, axis=1)
        s = s.reshape((s.shape[0], 1))
        return tmp / s

    def softmax_d(self, x):
        tmp = np.exp(x)
        return tmp / np.sum(tmp)


    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x/50))
        return y

    def sigmoid_d(self, x):
            return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    
    def ReLU(self, x):
        return x * (x > 0)

    def ReLU_d(self, x):
        return 1. * (x > 0)
    
    def LealyReLU(self, x):
        return (x * (x > 0) + 0.1* x * (x < 0))
    
    def LealyReLU_d(self, x):
        return (1. * (x > 0) - 0.1*(x<0))
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        return 1 - self.tanh(x) * self.tanh(x)
    

    def MSE(self, y_true, y_pred):
        return np.square(np.subtract(y_true, y_pred))/2
    
    def cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))



def main():
    pass


if __name__ == "__main__":main()