import numpy as np

import maelstrom


class Evaluator:
    def evaluate(self, forecast_reference_time, fcst, targets):
        raise NotImplementedError()

    def close(self):
        pass


class Verif(Evaluator):
    def __init__(
        self, filename, leadtimes, points, quantiles=None, attributes=dict(), sampling=1
    ):
        self.filename = filename

        if 0.5 not in quantiles:
            print(
                "Note: quantile=0.5 not in output. Determinsitic forecast not written to verif file"
            )

        self.sampling = sampling
        self.leadtimes = leadtimes
        self.points = points
        self.quantiles = quantiles

        kwargs = dict()
        if len(quantiles) > 1 or quantiles[0] != 0.5:
            kwargs["quantiles"] = quantiles
            self.write_quantiles = True
        else:
            self.write_quantiles = False

        self.file = maelstrom.output.VerifFile(
            filename,
            points,
            [i // 3600 for i in self.leadtimes],
            extra_attributes=attributes,
            **kwargs,
        )

        # A cache for the observations: valid_time -> observations
        self.obs_cache = set()

    def evaluate(self, forecast_reference_time, leadtime, fcst, targets):
        assert len(fcst.shape) == 3

        # Add observations
        curr_obs = np.reshape(
            targets[:: self.sampling, :: self.sampling, 0],
            [self.points.size()],
        )
        self.file.add_observations(forecast_reference_time + leadtime, curr_obs)
        self.obs_cache.add(curr_valid_time)
        # print("obs:", curr_obs)

        # Add determinsitic forecast
        if 0.5 in self.quantiles:
            I50 = self.quantiles.index(0.5)

            curr_fcst = fcst[..., I50]

            curr_fcst = np.reshape(
                curr_fcst[:: self.sampling, :: self.sampling],
                [len(self.leadtimes), self.points.size()],
            )
            self.file.add_forecast(forecast_reference_time, curr_fcst)
            # print("Fcst", i, np.nanmean(curr_fcst))

        # Add probabilistic forecast
        if self.write_quantiles:
            num_outputs = len(self.quantiles)
            curr_fcst = np.reshape(
                fcst[0, :, :: self.sampling, :: self.sampling, :],
                [len(self.leadtimes), self.points.size(), num_outputs],
            )
            self.file.add_quantile_forecast(forecast_reference_time, curr_fcst)
        self.file.sync()

    def close(self):
        self.file.write()


class Aggregator(Evaluator):
    def __init__(self, filename, leadtimes, loss):
        self.filename = filename
        self.leadtimes = leadtimes
        self.loss = loss
        with open(self.filename, "w") as file:
            file.write("unixtime leadtime obs fcst loss\n")

    def evaluate(self, forecast_reference_time, leadtime, fcst, targets):
        """
        Args:
            forecast_reference_time (float): Forecast reference time of forecasts
            fcst (np.array): 4D array of forecasts (y, x, output_variable)
            targets (np.array): 4D array of targets (y, x, target_variable)
        """
        assert len(fcst.shape) == 3
        assert len(targets.shape) == 3

        with open(self.filename, "a") as file:
            curr_loss = self.loss(targets, fcst)
            file.write(
                "%d %d %.5f %.5f %.5f\n"
                % (
                    forecast_reference_time,
                    leadtime // 3600,
                    np.nanmean(targets),
                    np.nanmean(fcst),
                    curr_loss,
                )
            )
