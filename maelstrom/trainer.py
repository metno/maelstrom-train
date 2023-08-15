import sys
import time
import collections

import numpy as np
import tensorflow as tf
import tqdm


def get(**kwargs):
    name = kwargs["type"].lower()
    args = {k: v for k, v in kwargs.items() if k not in ["type"]}
    if name == "keras":
        return Keras(**args)
    elif name == "trainer":
        return Trainer(**args)


class Trainer:
    """Object that trains a model

    Usage:
        trainer = maelstrom.trainer.Trainer()
        trainer.fit(model, optimizer, dataset, loss, epochs)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss,
        steps=1,
        callbacks=[],
        grad_type="mean",
        validation_frequency=1,
    ):
        """
        Args:
            repeat (int): How many steps
        """
        if grad_type not in ["median", "pseudo_median", "max", "mean", "random"]:
            raise ValueError("grad_type unknown")

        self.steps = steps
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks, add_history=True, model=self.model
        )
        self.grad_type = grad_type
        self.validation_frequency = validation_frequency

    def fit(self, data, epochs, validation_data=None, callbacks=[], steps_per_epoch=1):
        """Custom training loop

        Note that this function assumes that data contains enough data to finish all the epochs.
        That is, it should have epochs * steps_per_epoch batches. This means you will potentially
        have to repeat the dataset before passing it in.

        Args:
            data (tf.data): Dataset used for training
            epochs (int): Number of epochs to run for
            validation_data(tf.data): Optional validation dataset
            callbacks (list): Keras callbacks to call
            steps_per_epoch (int): Number of batches to process from data until one epoch is
                finished
        """
        # Assume that we run data to exhaustion and count epochs ourselves
        logs = {}
        callback_list = tf.keras.callbacks.CallbackList(
            callbacks, add_history=True, model=self.model
        )

        callback_list.on_train_begin(logs=logs)

        num_batches = steps_per_epoch * epochs

        s_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()

        timing = collections.defaultdict(lambda: 0)

        e_time = time.time()
        for batch, (x, y) in enumerate(data):
            s_time = time.time()
            read_time = s_time - e_time 
            epoch = batch // steps_per_epoch
            steps_since_last_epoch = batch % steps_per_epoch
            if steps_since_last_epoch == 0:
                # Starting a new epoch
                print(f"Epoch {epoch+1}/{epochs}")
                progbar = tf.keras.utils.Progbar(
                    steps_per_epoch
                )  # , stateful_metrics=['val_loss'])
                callback_list.on_epoch_begin(epoch, logs=logs)

            callback_list.on_train_batch_begin(batch, logs=logs)

            """Training step"""
            ss_time = time.time()
            loss_value, forward_time, backward_time = self.train_step(x, y)
            # forward_time = time.time() - ss_time

            # Moving window batch loss
            ss_time = time.time()
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            logs["loss"] = epoch_loss_avg.result()
            # backward_time = time.time() - ss_time

            # Instantaneous batch loss
            # logs["loss"] = loss_value

            callback_list.on_train_batch_end(batch, logs=logs)
            values = [(k, v) for k, v in logs.items()]

            prev_e_time = e_time
            e_time = time.time()
            batch_time = e_time - prev_e_time
            timing["read_time"] += read_time
            timing["forward_time"] += forward_time
            timing["backward_time"] += backward_time
            timing["count"] += 1
            timing["batch_time"] += batch_time
            timing["other_time"] += batch_time - read_time - forward_time - backward_time
            train_logs = logs
            num_batches = batch + 1

            val_logs = dict()
            if steps_since_last_epoch == steps_per_epoch - 1:
                # This is the end of the epoch
                if validation_data is not None:
                    """Validation"""
                    logs = None
                    count = 0
                    for batch, (x, y) in enumerate(validation_data):
                        callback_list.on_test_batch_begin(batch, logs=logs)

                        # logs = self.test_step(x, y)
                        curr_logs = self.model.test_on_batch(x=x, y=y, return_dict=True)
                        if logs is None:
                            logs = curr_logs
                        else:
                            for k, v in curr_logs.items():
                                logs[k] += v
                        count += 1

                        callback_list.on_test_batch_end(batch, logs=logs)
                    for k, v in logs.items():
                        val_logs["val_%s" % k] = v / count
                    values = [(k, v) for k, v in val_logs.items()]
                    # progbar.update(
                    #     num_batches + 1, values, True
                    # )  # This will update the progress bar graph.
                    # # progbar.add(0, values) # This will update the progress bar graph.
                else:
                    progbar.update(steps_since_last_epoch, values, True)

                e_time = time.time()
                callback_list.on_epoch_end(epoch, val_logs)
                print(f"Epoch {epoch} statistics:")
                for k,v in timing.items():
                    if k != "count":
                        val = v / timing["count"] * 1000
                        print(f"   {k}: {val:.2f} ms")
            else:
                progbar.update(steps_since_last_epoch, values, False)

        callback_list.on_train_end()

    def predict(self, dataset, **kwargs):
        return self.model.predict(dataset, **kwargs)

    def predict_on_batch(self, dataset, **kwargs):
        return self.model.predict_on_batch(dataset, **kwargs)

    def evaluate(self, datase, **kwargst):
        return self.model.evaluate(dataset, **kwargs)

    @tf.function
    def train_step(self, x, y):
        forward_time = 0
        backward_time = 0
        for i in range(self.steps):
            """Forward pass"""
            s_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                logits = self.model(x, training=True)
                if self.grad_type == "mean":
                    # loss_value = 0
                    loss_value = []
                    for q in range(y.shape[1]):
                        loss_value += [self.loss(y[:, q, ...], logits[:, q, ...])]
                    # loss_value /= y.shape[1]
                    # loss_value = self.loss(y[:, -1, ...], logits[:, -1, ...])
                else:
                    loss_value = [self.loss(y, logits)]
            e_time = time.time()
            forward_time += e_time - s_time

            # print("logits", logits.shape)
            # print(loss_value)
            # print("watched variables", tape.watched_variables())
            s_time = time.time()
            if self.grad_type == "mean":
                grads = tape.gradient(loss_value, self.model.trainable_weights)
            else:
                grads = None
                abs = 0
                for i in range(len(loss_value)):
                    curr = tape.gradient(loss_value[i], self.model.trainable_weights)
                    if i == 0:
                        grads = list()
                        for j in range(len(curr)):
                            grads += [list()]
                    for j, c in enumerate(curr):
                        grads[j] += [c]

                if self.grad_type == "median":
                    grads = self.get_median_grads(grads)
                elif self.grad_type == "pseudo_median":
                    grads = self.get_pseudo_median_grads(grads)
                elif self.grad_type == "max":
                    grads = self.get_max_grads(grads)
                elif self.grad_type == "mean":
                    grads = self.get_mean_grads(grads)
                elif self.grad_type == "random":
                    grads = self.get_random_grads(grads)
                # abs += tf.math.reduce_mean(tf.math.abs(grads[0]))
                """
                print()
                for g in grads:
                    print(g[:].numpy().flatten())
                    # print(g[:].flatten())
                # print(grads)
                """
            # print("grad", i, grads[i].numpy().flatten())
            # print("trainable_weights", self.model.trainable_weights)
            # print("grads", grads)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            backward_time += time.time() - s_time

        # print("ABS", abs)

        return tf.math.reduce_mean(loss_value), forward_time, backward_time

    @tf.function
    def test_step(self, x, y):
        pass

    @staticmethod
    def get_mean_grads(grads):
        output = list()
        for i in range(len(grads)):
            curr = tf.add_n(grads[i]) / len(grads[i])
            # if i == 0:
            #     print()
            #     print(tf.math.reduce_mean(curr).numpy())
            output += [curr]
        return output

    @staticmethod
    def get_median_grads(grads):
        output = list()
        params = list()
        # Loop over each layer
        for i in range(len(grads)):
            # print(grads[i][0].shape)
            curr = np.nan * np.zeros(
                [len(grads[i])] + list(grads[i][0].shape), np.float32
            )
            # loop over each output slice
            for j in range(len(grads[i])):
                # print(grads[i][j])
                curr[j, ...] = grads[i][j].numpy()
            c = tf.constant(np.median(curr, axis=0))
            params += [c]
            # params += [np.mean(curr, axis=0)]
            # params += [curr]
        output = params

        return output

    @staticmethod
    def get_pseudo_median_grads(grads):
        def compute(ar, func, mean):
            frac = tf.cast(func(grads[i][0], mean), np.float32)
            new_mean = frac * grads[i][0]
            N = len(grads[i])
            for j in range(1, N):
                curr = tf.cast(func(grads[i][j], mean), np.float32)
                new_mean += curr * grads[i][j]
                frac += curr
            frac /= N
            frac = tf.math.maximum(frac, 0.01)
            frac = tf.math.minimum(frac, 0.99)
            new_mean /= N
            new_mean /= frac
            return frac, new_mean

        output = list()
        # Loop over each layer
        for i in range(len(grads)):
            if 0:
                mean = tf.add_n(grads[i]) / len(grads[i])
                frac_above_mean, above_mean = compute(
                    grads[i], tf.math.greater_equal, mean
                )
                frac_below_mean, below_mean = compute(
                    grads[i], tf.math.less_equal, mean
                )

                # 2nd order polynomial
                a = below_mean
                c = (
                    ((mean - below_mean) - (above_mean - below_mean) * frac_below_mean)
                    / (-frac_above_mean)
                    / frac_below_mean
                )
                b = above_mean - below_mean - c
                curr = a + b * 0.5 + c * 0.25
            else:
                x1 = tf.add_n(grads[i]) / len(grads[i])
                q1, x0 = compute(grads[i], tf.math.less_equal, x1)
                _, x2 = compute(grads[i], tf.math.greater_equal, x1)
                q0, _ = compute(grads[i], tf.math.less_equal, x0)
                q2, _ = compute(grads[i], tf.math.less_equal, x2)

                # 2nd order polynomial
                a = x0
                c = ((x1 - x0) - (x2 - x0) * q1) / (-q2) / q1
                b = x2 - x0 - c
                q = 0.5
                d = q2 - q0
                d = tf.maximum(0.25, q2 - q0)

                # We want the median, but we need to normalize this such that the endpoints (q0, q2)
                # are set to (0, 1).
                q = (0.5 - q0) / (d)

                curr = a + b * q + c * q * q

            # First order polynomial
            # curr = below_mean + (0.5 / frac_below_mean) * (mean - below_mean)
            output += [curr]

        return output

    def get_max_grads(grads):
        for i in range(len(grads)):
            for j in range(len(grads[i])):
                if j == 0:
                    max = grads[i][j]
                else:
                    max = tf.maximum(max, grads[i][j])
            # TODO: Count number of positive, and keep mode direction.
            # grads[i] = tf.maximum(grads[i])
            grads[i] = max
        return grads

    @staticmethod
    def get_random_grads(grads):
        I = np.random.randint(len(grads[0]))
        for i in range(len(grads)):
            grads[i] = grads[i][I]
        return grads


class Keras:
    """Standard keras trainer (e.g model.fit)"""

    def __init__(self, model, optimizer, loss, callbacks=[]):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        """
        self._callbacks = callbacks
        self.callbacks = tf.keras.callbacks.CallbackList(
            self._callbacks, add_history=True, model=self.model
        )
        """

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)
