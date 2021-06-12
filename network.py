import numpy as np
from myargs import args
from keras.models import Model
from keras.regularizers import l2, l1
from keras.layers import Input, Dense, Dropout, multiply
from mil_nets.layer import Feature_pooling, Mil_Attention, Last_Sigmoid
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score
import tensorflow as tf

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def test_eval(model, test_set):
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch,1),dtype=float)
    test_acc = np.zeros((num_test_batch,1),dtype=float)
    for ibatch, batch in enumerate(test_set):
        arrs = batch[0]
        result = model.test_on_batch({'input':batch[0]}, {'fp':batch[1], 'recon':batch[0]})
        result = {out: result[i] for i, out in enumerate(model.metrics_names)}
        test_loss[ibatch] = result['loss']
        test_acc[ibatch] = result['fp_bag_accuracy']
    return np.mean(test_loss), np.mean(test_acc)

def predict_eval(model, test_set):
    num_test_batch = len(test_set)
    y_true = []
    y_pred = []
    for ibatch, batch in enumerate(test_set):
        arrs = batch[0]
        y = batch[1]
        result = model.predict_on_batch({'input': batch[0]})
        y_true.append(np.asscalar(np.mean(y)))
        y_pred.append(np.asscalar(result[0]))
    print(y_true)
    print(y_pred)
    return y_true, y_pred

def train_eval(model, train_set):
    num_train_batch = len(train_set)
    train_loss = np.zeros((num_train_batch,1),dtype=float)
    train_acc = np.zeros((num_train_batch,1),dtype=float)
    fp_loss = np.zeros((num_train_batch,1),dtype=float)
    recon_loss = np.zeros((num_train_batch,1),dtype=float)
    for ibatch, batch in enumerate(train_set):
        arrs = batch[0]
        result = model.train_on_batch({'input':batch[0]}, {'fp':batch[1], 'recon':batch[0]})
        result = {out: result[i] for i, out in enumerate(model.metrics_names)}
        train_loss[ibatch] = result['loss']
        train_acc[ibatch] = result['fp_bag_accuracy']
        fp_loss[ibatch] = result['fp_loss']
        recon_loss[ibatch] = result['recon_loss']
    return np.mean(train_loss), np.mean(train_acc), np.mean(fp_loss), np.mean(recon_loss)


class AMINN:
    @staticmethod
    def build_MINN_branch(h):
        fc1 = Dense(32, activation='relu', kernel_regularizer=l2(args.decay))(h)
        fc2 = Dense(32, activation='relu', kernel_regularizer=l2(args.decay))(fc1)
        fc3 = Dense(32, activation='relu', kernel_regularizer=l2(args.decay))(fc2)
        fc3 = Dropout(rate=args.do)(fc3)
        if args.pooling == 'att':
            alpha = Mil_Attention(L_dim=64, output_dim=1, kernel_regularizer= l2(args.decay), name='alpha',
                                  use_gated=False)(fc3)
            x_mul = multiply([alpha, fc3], name='mul')
            fp = Last_Sigmoid(output_dim=1, name='fp')(x_mul)
        else:
            fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(args.decay), pooling_mode = args.pooling, name='fp')(fc3)
        return fp

    @staticmethod
    def build_AE_brach(inputs):
        dimensions = inputs[0].shape[0]
        fc1 = Dense(64, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(inputs)
        fc2 = Dense(32, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(fc1)
        fc3 = Dense(16, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(fc2)
        h = Dense(8, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3), name='hidden')(fc3)
        fc4 = Dense(16, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(h)
        fc5 = Dense(32, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(fc4)
        fc6 = Dense(64, activation='relu', kernel_regularizer=l2(args.decay), activity_regularizer=l1(args.decay*1e-3))(fc5)
        out = Dense(dimensions, activation='sigmoid', name='recon')(fc6)
        return out, h

    @staticmethod
    def build(inputs):
        dimensions = inputs[0][0].shape[1]
        inputs = Input(shape=(dimensions,), dtype='float32', name='input')
        AE_branch, h = AMINN.build_AE_brach(inputs)
        MINN_branch = AMINN.build_MINN_branch(h)
        model = Model(inputs = [inputs], outputs = [MINN_branch, AE_branch], name= 'AEMIN')
        plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
        return model
