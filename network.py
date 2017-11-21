"""Class that represents the network to be evolved."""
import random
import logging
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from utils import *

early_stopper = EarlyStopping(patience=5)

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, parameter_choices=None):
        """Initialize our network.

        Args:
            parameter_choices (dict): Parameters for the network, includes:
                neurons_per_layer (list): [64, 128, 256, 512, 1028]
                number_of_layers (list): [1, 2, 3, 4, 5]
                activation (list): ['relu', 'elu']
                dropout (list): [0.2, 0.4, 0.6]
                optimizer (list): ['rmsprop', 'adam']
        """

        self.accuracy = 0.
        self.loss = 0.
        self.model = None
        self.parameter_choices = parameter_choices
        self.network = {}  # (dic): represents MLP network parameters


    def get_parameter(self, parameter):
        return self.network[parameter]

    def set_parameter(self, parameter, new_value):
         self.network[parameter] = new_value

    def correct_parameter(self, parameter):
        print("parameter start: " + str(self.network[parameter]))

        while self.network['number_of_layers'] > len(self.network[parameter]):
            self.network[parameter].append(self.network[parameter][-1])   #Fill neurons_per_layer with it's last value

        if self.network['number_of_layers'] < len(self.network[parameter]):
            self.network[parameter] = self.network[parameter][:self.network['number_of_layers']]     #Discard the extra values

        print("parameter end: " + str(self.network[parameter]))

    def is_consistent(self):
        if type(self.network['neurons_per_layer']) is int or type(self.network['dropout_per_layer']) is float:
            print("hay un type int o float")
            return False

        if self.network['number_of_layers'] != len(self.network['neurons_per_layer']) or self.network['number_of_layers'] != len(self.network['dropout_per_layer']):
            print("no coinciden dimensiones")
            return False

        return True

    def create_random(self):
        """Create a random network."""
        number_of_layers = random.choice(self.parameter_choices['number_of_layers'])
        neurons_per_layer = []
        dropout_per_layer = []
        self.network['number_of_layers'] = number_of_layers

        for i in range(number_of_layers):
            neurons_per_layer.append(random.choice(self.parameter_choices['neurons_per_layer']))
            dropout_per_layer.append(random.choice(self.parameter_choices['dropout_per_layer']))

        self.network['neurons_per_layer'] = neurons_per_layer
        self.network['dropout_per_layer'] = dropout_per_layer
        self.network['optimizer'] = random.choice(self.parameter_choices['optimizer'])
        self.network['activation'] = random.choice(self.parameter_choices['activation'])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def get_mnist(self):
        """Retrieve the MNIST dataset and process the data."""
        # Set defaults.
        nb_classes = 10
        batch_size = 128
        input_shape = (784,)

        # Get the data.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_mnist()

        self.model = self.compile_model(nb_classes, input_shape)

        self.model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=0,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopper])

        score = self.model.evaluate(x_test, y_test, verbose=0)

        self.loss, self.accuracy = score[0], score[1]  # 1 is accuracy. 0 is loss.

        return self.loss, self.accuracy

    def compile_model(self, nb_classes, input_shape):
        """Compile a sequential model.

        Args:
            network (dict): the parameters of the network
            nb_classes (int): number of dataset nb_classes
            input_shape (int):

        Returns:
            a compiled network.

        """

        # Get our network parameters.
        number_of_layers = self.network['number_of_layers']
        neurons_per_layer = self.network['neurons_per_layer']
        dropout_per_layer = self.network['dropout_per_layer']
        activation = self.network['activation']
        optimizer = self.network['optimizer']

        model = Sequential()

        if type(neurons_per_layer) is int:
            self.network['neurons_per_layer'] = [neurons_per_layer]

        if type(dropout_per_layer) is float:
            self.network['dropout_per_layer'] = [dropout_per_layer]


        if number_of_layers != len(self.network['neurons_per_layer']):
            self.network['neurons_per_layer'] = correct_neurons_per_layer(self.network)

        if number_of_layers != len(self.network['dropout_per_layer']):
            self.network['dropout_per_layer'] = correct_dropout_per_layer(self.network)

        print("number of layers: " + str(number_of_layers))
        print("neurons: " + str(self.network['neurons_per_layer']) + " with " + activation)
        print("dropout: " + str(self.network['dropout_per_layer']))

        # Add each layer.
        for i in range(len(self.network['neurons_per_layer'])):
            # Need input shape for first layer.
            if i == 0:
                model.add(Dense(self.network['neurons_per_layer'][i], activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(self.network['neurons_per_layer'][i], activation=activation))

        model.add(Dropout(self.network['dropout_per_layer'][i]))

        # Output layer.
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def print_network(self):
        """Print out a network."""
        plot_model(self.model, to_file='model.png', show_shapes=True)
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
        logging.info("Network loss: %.2f%%" % (self.loss))
