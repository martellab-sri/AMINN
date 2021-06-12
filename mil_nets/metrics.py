from keras import backend as K
from keras import losses


def bag_loss(y_true, y_pred):
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def bag_accuracy(y_true, y_pred):
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_mse(x, x_recon):
    loss = K.mean(losses.mean_squared_error(x, x_recon), axis=-1)
    return loss
