import numpy as np
import xarray as xr
import tensorflow as tf

import maelstrom

with_horovod = maelstrom.check_horovod()
if with_horovod:
    import horovod.tensorflow as hvd


class Evaluator:
    def evaluate(self, forecast_reference_time, fcst, targets):
        raise NotImplementedError()

    def sync(self):
        pass

    def write(self):
        pass

    def close(self):
        pass


class Verif(Evaluator):
    def __init__(
        self, filename, leadtimes, lats, lons, elevs, quantiles=None, attributes=dict(), sampling=1
    ):
        self.filename = filename
        self.leadtimes = leadtimes
        self.lats = lats
        self.lons = lons
        self.elevs = elevs
        self.quantiles = quantiles
        self.attributes = attributes
        self.sampling = sampling

        # Save values as a list of (frt, leadtime, fcst, obs) tuples
        # fcst have shape (y, x, output) and obs have shape (y, x)
        self.values = list()


    def evaluate(self, forecast_reference_time, leadtime, fcst, targets):
        assert len(fcst.shape) == 3
        assert len(targets.shape) == 3

        # Add observations
        curr_obs = targets[::self.sampling, :: self.sampling, 0]
        curr_fcst = fcst[::self.sampling, :: self.sampling, :]
        values = (forecast_reference_time, leadtime, curr_fcst, curr_obs)
        self.values += [values]

    def sync(self):
        self.values = hvd.allgather_object(self.values)
        self.values = sum(allvalues, [])

    def write(self):
        frts = np.sort(np.unique([i[0] for i in self.values]))
        leadtimes = np.sort(np.unique([i[1] for i in self.values]))

        T = len(frts)
        LT = len(leadtimes)
        lats = self.lats[::self.sampling, ::self.sampling].flatten()
        lons = self.lons[::self.sampling, ::self.sampling].flatten()
        elevs = self.elevs[::self.sampling, ::self.sampling].flatten()
        L = len(lats)
        Q = len(self.quantiles)

        fcst = np.nan * np.zeros([T, LT, L, Q], np.float32)
        obs = np.nan * np.zeros([T, LT, L], np.float32)

        # Loop over all stored data and put into fcst/obs arrays
        for t in range(len(self.values)):
            curr_frt, curr_leadtime, curr_fcst, curr_obs = self.values[t]
            It = np.where(frts == curr_frt)[0][0]
            Ilt = np.where(leadtimes == curr_leadtime)[0][0]
            curr_fcst = np.reshape(curr_fcst, [L, Q])
            curr_obs = curr_obs.flatten()
            fcst[It, Ilt, :, :] = curr_fcst
            obs[It, Ilt, :] = curr_obs

        data_vars = dict()
        data_vars["time"] = frts
        data_vars["leadtime"] = leadtimes // 3600
        data_vars["lat"] = (("location",), lats)
        data_vars["quantile"] = self.quantiles
        data_vars["lon"] = (("location", ), lons)
        data_vars["location"] = np.arange(L)
        data_vars["x"] = (("time", "leadtime", "location", "quantile"), fcst)
        if 0.5 in self.quantiles:
            I = self.quantiles.index(0.5)
            data_vars["fcst"] = (("time", "leadtime", "location"), fcst[:, :, :, I])
        data_vars["obs"] = (("time", "leadtime", "location"), obs)
        coords = dict()
        dataset = xr.Dataset(data_vars, coords, self.attributes)
        dataset.to_netcdf(self.filename)


class Aggregator(Evaluator):
    def __init__(self, filename, leadtimes, loss, metrics):
        self.filename = filename
        self.leadtimes = leadtimes
        self.loss = loss
        self.metrics = metrics
        # Store data as a list of tuples. Each tuple contains the metadata and loss values
        self.values = list()

    def evaluate(self, forecast_reference_time, leadtime, fcst, targets):
        """
        Args:
            forecast_reference_time (float): Forecast reference time of forecasts
            fcst (np.array): 4D array of forecasts (y, x, output_variable)
            targets (np.array): 4D array of targets (y, x, target_variable)
        """
        assert len(fcst.shape) == 3
        assert len(targets.shape) == 3

        # Inputs are numpy arrays, but many of the loss functions need tf tensors
        fcst = tf.cast(fcst, tf.float32)
        targets = tf.cast(targets, tf.float32)

        curr_loss = self.loss(targets, fcst)
        curr_metrics = list()
        for metric in self.metrics:
            curr_metrics += [metric(targets, fcst)]
        meanfcst = np.mean(fcst)
        meantarget = np.mean(targets)
        values = (forecast_reference_time, leadtime, meantarget, meanfcst, curr_loss, curr_metrics)
        self.values += [values]

    def sync(self):
        allvalues = hvd.allgather_object(self.values)
        self.values = sum(allvalues, [])

    def write(self):
        with open(self.filename, "w") as file:
            file.write("unixtime leadtime obs fcst loss")
            for metric in self.metrics:
                file.write(f" {metric.__name__}")
            file.write("\n")
            for forecast_reference_time, leadtime, meantarget, meanfcst, loss_value, metric_values in self.values:
                file.write(
                    "%d %d %.5f %.5f %.5f"
                    % (
                        forecast_reference_time,
                        leadtime // 3600,
                        meantarget,
                        meanfcst,
                        loss_value,
                    )
                )
                for metric_value in metric_values:
                    file.write(f" %.5f" % metric_value)
                file.write("\n")
