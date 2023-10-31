import tensorflow as tf
import tensorflow.keras.backend as K

import maelstrom


def get(config, quantile_levels=None):
    """Factory method for loss functions

    Args:
        config (dict): Settings for loss function
        quantile_levels (list): List of quantile levels
    """
    name = config["type"]
    args = {k:v for k,v in config.items() if k != "type"}
    if name == "mae":
        loss = maelstrom.loss.mae
    elif name == "mae_prob":
        loss = maelstrom.loss.mae_prob
    elif name == "sharpness":
        loss = maelstrom.loss.sharpness
    elif name == "reliability":
        loss = maelstrom.loss.reliability
    else:
        if name == "quantile_score":
            loss = lambda x, y: maelstrom.loss.quantile_score(x, y, quantile_levels, **args)
        elif name == "quantile_score_prob":
            loss = lambda x, y: maelstrom.loss.quantile_score_prob(x, y, quantile_levels, **args)
        elif name == "within":
            index = 0
            if quantile_levels is not None:
                index = len(quantile_levels) // 2
            loss = lambda x, y: maelstrom.loss.within(x, y, index, **args)
        else:
            raise NotImplementedError(f"Unknown loss function {name}")

        # This is needed, otherwise it will show up as <lambda> if used as a metric in keras.fit
        loss.__name__ = name

    return loss


def quantile_score(y_true, y_pred, quantile_levels, trim=None):
    qtloss = 0
    for i, quantile in enumerate(quantile_levels):
        err = y_true[..., 0] - y_pred[..., i]
        if trim is not None:
            err = err[..., trim:-trim, trim:-trim]
        qtloss += (quantile - tf.cast((err < 0), tf.float32)) * err
    return K.mean(qtloss) / len(quantile_levels)


def quantile_score_prob(y_true, y_pred, quantile_levels, trim=None):
    """

    Weighted version:
    return K.mean((qtloss0 + qtloss1 + qtloss2) / (1 + y_true_std))
    """
    y_true_mean = y_true[..., 0]
    y_true_std = y_true[..., 1]

    d = y_true_std * 1.28155  # scipy.stats.norm.ppf(0.1)
    weighted = False

    qtloss = 0
    for i, quantile in enumerate(quantile_levels):
        if weighted:
            err = y_true_mean - y_pred[..., i]
            curr = (quantile_levels[i] - tf.cast((err < 0), tf.float32)) * err
            qtloss += curr / (1 + y_true_std)
        else:
            s = 0.6 * (K.log(quantile)*2 - K.log((1-quantile)*2))
            err = y_true_mean + s - y_pred[..., i]
            qtloss += (quantile_levels[i] - tf.cast((err < 0), tf.float32)) * err

    if trim is not None:
        qtloss = qtloss[..., trim:-trim, trim:-trim]
    return K.mean(qtloss / len(quantile_levels))


def mae(y_true, y_pred):
    return K.mean(K.abs(y_true[..., 0] - y_pred[..., 0]))


def mae_prob(y_true, y_pred):
    y_true_mean = y_true[..., 0]
    y_true_std = y_true[..., 1]
    diff = K.abs(y_true_mean - y_pred[..., 0])
    return K.mean(diff + 0.8 * y_true_std * K.exp(-1.4 / y_true_std * diff))


def meanfcst(y_true, y_pred):
    return K.mean(y_pred)
    # num_leadtimes = y_true.shape[3]
    # # return K.mean(y_pred[:, :, :, 0:num_leadtimes])


def meanobs(y_true, y_pred):
    return K.mean(y_true)

def sharpness(y_true, y_pred):
    """Width of outer quantile_levels"""
    return K.mean(y_pred[..., -1] - y_pred[..., 0])

def reliability(y_true, y_pred):
    """Fraction of targets that fall within the outer quantile_levels"""
    return K.mean(tf.cast(y_pred[..., 0] < y_true[..., 0], tf.float32) * tf.cast(y_pred[..., -1] > y_true[..., 0], tf.float32))

def within(y_true, y_pred, index=0):
    err = K.abs(y_true[..., 0] - y_pred[..., index])
    return K.mean(err > 3)
