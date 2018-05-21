import pandas as pd
import numpy as np
import tensorflow as tf
import io
from sklearn.utils import shuffle
import os
import tensorflow as tf
import pandas as pd
import itertools
import numpy as np
import os
from time import time
import sys

class gc:
    pass

def set_globals(g):
    g.eval_size = 0
    g.steps = 10
    g.step_sets = 2
    g.tags= ['a']
    g.tag_labels = [i.replace(' ','_').replace('&','_') for i in g.tags]
    g.loctags = ['b']
    g.loctag_labels = [i.replace(' ','_').replace('&','_') for i in g.loctags]
    g.model_layers = {'mlturnmodel1ctest2d': [500, 100, 30, 100]}
    g.hidden_layers = g.model_layers[g.model_name]
    g.model_dir_name = g.model_name

def define_labels(g):
    g.labels = ['t']
    g.turn_tf_num = tf.feature_column.numeric_column('t')

def define_linear_features(g):
    g.linear_feature_names = ['s', 'i']
    g.linear_features_tf = []
    for f in g.linear_feature_names:
        g.linear_features_tf.append(tf.feature_column.numeric_column(f))

def define_sparse_features(g):
    g.sparse_feature_names = []
    for t in g.tag_labels:
        g.sparse_feature_names.append(t+'0')
    for t in g.loctag_labels:
        g.sparse_feature_names.append(t+'1')
    g.sparse_features_tf = []
    for f in g.sparse_feature_names:
        g.sparse_features_tf.append(tf.feature_column.numeric_column(f))

def init(g):
    set_globals(g)
    define_labels(g)
    define_linear_features(g)
    define_sparse_features(g)
    g.all_feature_names = g.linear_feature_names + g.sparse_feature_names
    g.all_features_tf = g.linear_features_tf + g.sparse_features_tf
    print('\nlabels:\n', len(g.labels), ',', g.labels)
    print('\nlinear features:\n', len(g.linear_feature_names), ',', g.linear_feature_names)
    print('\nsparse features:\n', len(g.sparse_feature_names), ',', g.sparse_feature_names)
    g.dnn_model = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=g.linear_features_tf,
        dnn_feature_columns=g.sparse_features_tf,
        dnn_hidden_units=g.hidden_layers, dnn_dropout=0.3,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.01,
            l1_regularization_strength=0.01,
        ), label_dimension=1,
        model_dir=g.model_dir_name)

def prepare_data(g):
    pklfname = './'+g.model_name+'.pkl'
    if os.path.isfile(pklfname):
        print('reading pkl: '+pklfname)
        df0 = pd.read_pickle(pklfname)
    rows = df0.shape[0]
    print('rows '+str(rows))
    dfin = df0.iloc[0:rows - g.eval_size]
    print('len dfin '+str(len(dfin)))
    g.train_data = dfin.values
    print('train data values size ', g.train_data.shape)
    print(str(g.train_data[0]))
    g.train_labels = dfin['t'].values
    print(str(g.train_labels[0]))

def train(g):
    print('train')
    x = {g.all_feature_names[i]: g.train_data[:,i]
         for i in range(len(g.all_feature_names))}
    y = g.train_labels
    train_input = tf.estimator.inputs.numpy_input_fn(
        x = x, y = y, shuffle=True, num_threads=1)
    time0 = time()
    g.dnn_model.train(input_fn=train_input, steps=g.steps)
    print('\nFIT TIME: ', time() - time0)
    return g

if __name__ == '__main__':
    g = gc()
    g.model_name = 'mlturnmodel1ctest2d'
    time0 = time()
    g.train = True
    init(g)
    prepare_data(g)
    if g.train:
        for i in range(g.step_sets):
            g = train(g)
