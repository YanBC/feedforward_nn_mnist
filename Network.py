import numpy as np

from config import CONFIG


# activation function
#
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# cost function
# 
def cost(y_true, y_pred):
    cost = np.square(y_true - y_pred)
    cost = np.mean(cost)
    return cost

def cost_prime(y_true, y_pred):
    return y_pred - y_true


# accuracy function
#
def accuracy(y_true, y_pred):
    indice_true = np.argmax(y_true, axis=1)
    indice_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(indice_true==indice_pred)
    return correct/len(indice_true)


# a function to transform a label vector to an one-hot array
#
# Inputs:
# @labels: a numpy array of shape (num_of_labels,)
# @num_class: an integer specifying the number of classes
#
# Outputs:
# @ret: a numpy array of shape (num_of_labels, num_class)
#
def one_hot(labels, num_class):
    ret = np.zeros((labels.shape[0], num_class))
    ret[np.arange(labels.shape[0]), labels] = 1
    return ret



class Network(object):
    # initialize Network
    #
    # Inputs:
    # @size_layers: a list of integers representing numbers of neurons in each layers
    #
    def __init__(self, size_layers):
        # for book keeping purpose
        self.num_layers = len(size_layers)
        self.input_shape = (None, size_layers[0])
        self.output_shape = (None, size_layers[-1])

        # create containers for weights and bias
        self.weights = [None] * (self.num_layers -1)
        self.bias = [None] * (self.num_layers -1)

        # randomly initialize all weights and bias
        for i in range(self.num_layers - 1):
            self.weights[i] = np.random.randn(size_layers[i], size_layers[i+1])
            self.bias[i] = np.random.randn(1, size_layers[i+1])

    # predit method
    #
    # Inputs:
    # @x: a numpy array of shape self.input_shape
    #
    # Outputs:
    # @ret: a numpy array of shape self.output_shape
    #
    def predit(self, x):
        y = x.copy()
        for i in range(self.num_layers - 1):
            y = np.matmul(y, self.weights[i]) + self.bias[i]
            y = sigmoid(y)
        indice = np.argmax(y, axis=1)
        ret = np.zeros(y.shape)
        ret[np.arange(ret.shape[0]), indice] = 1
        return ret

    # train method
    # 
    # Inputs:
    # @data: a dictionary of length 4 and has keywords: 'training_images', 'training_labels', 'test_images', 'test_labels'
    #
    def train(self, data):
        # get data
        train_x = data['training_images']
        train_y = data['training_labels']
        val_x = data['test_images']
        val_y = data['test_labels']

        # for each epoch
        for epoch in range(CONFIG['epochs']):
            # print message
            print('Epoch #%d ...' % epoch)

            # shuffle the data
            if CONFIG['shuffle']:
                s = np.random.randint(CONFIG['shuffle_seed_range'])
                np.random.seed(s)
                np.random.shuffle(train_x)
                np.random.seed(s)
                np.random.shuffle(train_y)

            # for each batch
            total_steps = CONFIG['train_set_size'] // CONFIG['batch_size']
            for step in range(total_steps):
                mini_batch_x = train_x[step * CONFIG['batch_size']:(step + 1) * CONFIG['batch_size'], :]
                mini_batch_y = train_y[step * CONFIG['batch_size']:(step + 1) * CONFIG['batch_size'], :]
                
                # feed-forward
                z = [None] * len(self.bias)
                a = [None] * len(self.bias)
                z[0] = np.matmul(mini_batch_x, self.weights[0]) + self.bias[0]
                a[0] = sigmoid(z[0])
                for i in range(1, len(self.bias)):
                    z[i] = np.matmul(a[i-1], self.weights[i]) + self.bias[i]
                    a[i] = sigmoid(z[i])

                # calculate cost and accuracy
                c = cost(mini_batch_y, a[-1])
                acc = accuracy(mini_batch_y, a[-1])
                # print('Step #%d/%d: cost:%.2f, acc:%.2f' %(step, total_steps, c, acc))

                # calculate gradient
                delta = [None] * len(self.bias)
                delta[-1] = cost_prime(mini_batch_y, a[-1]) * sigmoid_prime(z[-1])
                for i in range(len(delta) - 1):
                    atLayer = -2 - i
                    delta[atLayer] = np.matmul(delta[atLayer + 1], np.transpose(self.weights[atLayer + 1])) * sigmoid_prime(z[atLayer])

                partial_bias = [np.mean(i, axis=0) for i in delta]
                a_previous = [mini_batch_x, *a[0:-1]]
                a_previous = [np.transpose(i) for i in a_previous]
                partial_weights = [np.matmul(i,j) / CONFIG['batch_size'] for i,j in zip(a_previous, delta)]

                # clip partial gradients
                # partial_weights = [np.clip(i, a_min=-0.5, a_max=0.5) for i in partial_weights]
                # partial_bias = [np.clip(i, a_min=-0.5, a_max=0.5) for i in partial_bias]

                # apply gradient
                eta = CONFIG['learning_rate']
                # if step > total_steps / 4:
                #     eta = CONFIG['learning_rate'] / 2
                # if step > total_steps / 2:
                #     eta = CONFIG['learning_rate'] / 4
                # if step > total_steps * 3 / 4:
                #     eta = CONFIG['learning_rate'] / 8
                self.weights = [i - eta * j for i,j in zip(self.weights, partial_weights)]
                self.bias = [i - eta * j for i,j in zip(self.bias, partial_bias)]

            # run validation set
            train_pred = self.predit(train_x)
            train_cost = cost(train_y, train_pred)
            train_acc = accuracy(train_y, train_pred)
            # print('Train: cost:%.7f, acc:%.4f' %(train_cost, train_acc))          
            val_pred = self.predit(val_x)
            val_cost = cost(val_y, val_pred)
            val_acc = accuracy(val_y, val_pred)
            print('Train: cost:%.7f, acc:%.4f;\tValidation: cost:%.7f, acc:%.4f' %(train_cost, train_acc, val_cost, val_acc))









