from keras import backend as K


def max_pooling(x):
    output = K.max(x, axis=0, keepdims=True)
    return output


def mean_pooling(x):
    output = K.mean(x, axis=0, keepdims=True)
    return output


def LSE_pooling(x):
    output = K.log(K.mean(K.exp(x), axis=0, keepdims=True))
    return output


def choice_pooling(x, pooling_mode):
    if pooling_mode == 'max':
        return max_pooling(x)
    if pooling_mode == 'lse':
        return LSE_pooling(x)
    if pooling_mode == 'ave':
        return mean_pooling(x)
