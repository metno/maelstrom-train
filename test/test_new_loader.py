from __future__ import print_function
import unittest
import collections
import numpy as np
import time

import maelstrom.loader


class Test(unittest.TestCase):
    def test_loader_1(self):
        # filename = "test/files/loader_1.yml"
        # config  = maelstrom.load_yaml(filename)["loader"]
        config = {
                  "filenames": [ "data_b/air_temperature/5TB/2020030*T*Z.nc"],
                  # "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
                  "normalization": "test/files/normalization.yml",
                  "predict_diff": True,
                  "limit_predictors": ["air_temperature_2m", "precipitation_amount", "altitude"],
                  "patch_size": 256,
                  "limit_leadtimes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  "extra_features": [{"type": "x"}],
                  }
        loader = maelstrom.new_loader.Loader(**config)
        dataset = loader.get_dataset(4)
        count = 0
        s_time = time.time()
        for q in dataset:
            print("Sample", count, q[0].shape)
            count += 1
        print("Total loading time", time.time() - s_time)
        maelstrom.util.print_memory_usage()
        print(loader.logger)
        return
        print(loader.num_leadtimes)
        predictors, targets = loader.parse_file("test/files/air_temperature/5GB/20200301T03Z.nc", [0])
        print(predictors.shape, targets.shape)
        return

        dataset = loader.get_dataset()
        count = 0
        for p, t in dataset:
            p = p.numpy()
            t = t.numpy()
            self.assertEqual(p.shape, (1, 5, 6, 4, 15))
            self.assertEqual(t.shape, (1, 5, 6, 4, 1))
            tol = 5
            if count == 0:
                # predictor 4 is cloud_area_fraction
                # ncdump -f c -v predictors 20200301T03Z.nc | grep "predictors(0,2,3,4)"
                expected_predictor = 0.99614
                # normalization: 0.6884219621618589, 0.40055377741854187
                expected_predictor = (expected_predictor - 0.6884219621618589) / 0.40055377741854187
                self.assertAlmostEqual(p[0, 0, 2, 3, 4], expected_predictor, tol)

                # Check normalization of x feature
                np.testing.assert_array_almost_equal(p[0, 0, 2, :, 14], (-1.34164079, -0.4472136 ,  0.4472136 , 1.34164079), tol)

                # ncdump -f c -v target_mean 20200301T03Z.nc | grep "target_mean(0,2,3)"
                expected_target = 0.5384477

                # ncdump -f c -v predictors 20200301T03Z.nc | grep "predictors(0,2,3,2)"
                expected_target -= 0.5564111
                self.assertAlmostEqual(t[0, 0, 2, 3, 0], expected_target, tol)
                # self.assertAlmostEqual(p, loader[0][0])
            count += 1

        # loader[0]
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
