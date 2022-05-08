import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from matplotlib import pyplot as plt
import random
from keras import optimizers, Model, metrics, backend, losses
from keras.layers import Dense, Activation, Dropout, Input
import os
import itertools
from tabulate import tabulate
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as backend_keras

# gpu_options = tf.GPUOptions()
# config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

cwd = os.getcwd()  # Get the current working directory
path_save = cwd + '\\Database\\'

# List of subjects
subject_names = ['001']
# List of exercises
exercices_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

num_classes = 5

def get_data(num_classes):

    DB_X = np.load(path_save + "DB_X.npy")
    DB_Y = np.load(path_save + "DB_Y.npy")

    p = np.array([i for i in range(DB_X.shape[0])])
    random.shuffle(p)

    n = DB_X.shape[0]
    n_sample_train = int(n * 0.8)

    train_X = DB_X[p][:n_sample_train]
    train_X = np.transpose(train_X, [0, 2, 1])
    train_X = train_X.reshape([train_X.shape[0], -1])

    test_X = DB_X[p][n_sample_train:]
    test_X = np.transpose(test_X, [0, 2, 1])
    test_X = test_X.reshape([test_X.shape[0], -1])

    train_Y = DB_Y[p][:n_sample_train]
    test_Y = DB_Y[p][n_sample_train:]

    train_Y = np_utils.to_categorical(train_Y, num_classes)
    test_Y = np_utils.to_categorical(test_Y, num_classes)

    return train_X, test_X, train_Y, test_Y

def rank_and_display(hyperparameters_list, accuracy_list):

    # Sort metric and generation
    idx = np.argsort(accuracy_list)
    hyperparameters_list_sorted = np.array(hyperparameters_list)[idx].tolist()
    hyperparameter_best = hyperparameters_list_sorted[-1]

    accuracy_list_sorted = np.array(accuracy_list)[idx].tolist()

    df = pd.DataFrame.from_dict(hyperparameters_list_sorted)
    df["Accuray"] = accuracy_list_sorted

    print(tabulate(df, headers='keys', tablefmt='psql'))

    return hyperparameter_best

def get_hyperparameters(num_classes):
    choices = dict(input_shape=[450], num_classes=[5], learning_rate=[0.1, 0.01, 0.001],
                   layers_units=[[64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128], [256, 256, 256]],
                   activation_function=['relu', 'sigmoid', 'tanh'])

    keys, values = zip(*choices.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    df = pd.DataFrame.from_dict(combs)

    print(tabulate(df, headers="keys", tablefmt="psql"))

    return combs

train_X, test_X, train_Y, test_Y = get_data(num_classes)

hyper = {'input_shape': 450, 'num_classes': 5, 'learning_rate': 0.1,
         'layers_units': [64, 64], 'activation_function': 'sigmoid'}

max_epoch = 20

def MLP(hyper):

    num_classes = hyper['num_classes']
    input_shape = hyper['input_shape']
    learning_rate = hyper['learning_rate']
    layers_units = hyper['layers_units']
    activation_function = hyper['activation_function']

    #Create graph and session
    tf.reset_default_graph()
    backend_keras.clear_session()

    graph = tf.get_default_graph()
    sess = tf.Session()

    backend_keras.set_session(sess)

    # Input layers
    x = Input(shape=(input_shape,), name="X")
    l = x


    for i in range(0, len(layers_units)):

        l = Dense(units=layers_units[i], name="Dense_" + str(i))(l)
        l = Activation(activation=activation_function, name="Activation_dense_" + str(i))(l)

    y_ = Dense(units=num_classes, name="output", activation="softmax")(l)

    #Scope
    loss = losses.categorical_crossentropy
    # "metric" is a measure that we use to evaluate how good the model is performing
    metric = metrics.categorical_accuracy
    # The optimizer used is a "stochastic gradient descent" (sgd) because it is faster than a regular gradient descent
    optimizer = optimizers.sgd()

    #Compile
    model = Model(inputs=[x], outputs=[y_])
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.summary()

    return model

combs = get_hyperparameters(num_classes=5)

hyper_accuracy_list = []

for hyper in combs:

    model = MLP(hyper)

    loss_hist = []
    accuracy_hist = []

    for i in range(max_epoch):

        # Training
        model.fit(x=train_X, y=train_Y, epochs=1, verbose=0)

        # Evaluation of the model
        loss, accuracy = model.evaluate(x=train_X, y=train_Y, verbose=0)

        loss_hist.append(loss)
        accuracy_hist.append(accuracy)

        print("Loss: " + str(loss) + " - Accuracy: " + str(accuracy))

    # Get the last accuracy for the current training (maybe improved with max in accuracy_hist
    hyper_accuracy_list.append(accuracy)

hyperparameter_best = rank_and_display(hyperparameters_list=combs, accuracy_list=hyper_accuracy_list)

model = MLP(hyperparameter_best)

loss_hist = []
accuracy_hist = []

for i in range(max_epoch):

    # Training
    model.fit(x=train_X, y=train_Y, epochs=1, verbose=0)

    # Evaluation of the model
    loss, accuracy = model.evaluate(x=test_X, y=test_Y, verbose=0)

    loss_hist.append(loss)
    accuracy_hist.append(accuracy)

    print("Loss: " + str(loss) + " - Accuracy: " + str(accuracy))
