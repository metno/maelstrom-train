from __future__ import print_function
import unittest
import collections
import numpy as np

import maelstrom.loader


class Test(unittest.TestCase):
    def test_loader_1(self):
        # filename = "test/files/loader_1.yml"
        # config  = maelstrom.load_yaml(filename)["loader"]
        config = {"type": "file",
                  "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
                  "normalization": "test/files/normalization.yml",
                  "predict_diff": True,
                  "extra_features": [{"type": "x"}],
                  "to_gpu": False,
                  }
        loader = maelstrom.new_loader.get(config)

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

    def test_patch(self):
        # filename = "test/files/loader_1.yml"
        # config  = maelstrom.load_yaml(filename)["loader"]
        config = {"type": "file",
                  "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
                  "normalization": "test/files/normalization.yml",
                  "predict_diff": True,
                  "patch_size": 2,
                  "extra_features": [{"type": "x"}],
                  }
        loader = maelstrom.new_loader.get(config)

        dataset = loader.get_dataset()
        count = 0
        for p, t in dataset:
            p = p.numpy()
            t = t.numpy()
            self.assertEqual(p.shape, (1, 5, 2, 2, 15))
            self.assertEqual(t.shape, (1, 5, 2, 2, 1))
            tol = 5
            print(count, p[0, 0, :, :, 14])
            if count == 3:
                # predictor 4 is cloud_area_fraction
                # ncdump -f c -v predictors 20200301T03Z.nc | grep "predictors(0,2,3,4)"
                expected_predictor = 0.99614
                # normalization: 0.6884219621618589, 0.40055377741854187
                expected_predictor = (expected_predictor - 0.6884219621618589) / 0.40055377741854187
                # self.assertAlmostEqual(p[0, 0, 0, 1, 4], expected_predictor, tol)

                # Check normalization of x feature
                np.testing.assert_array_almost_equal(p[0, 0, 0, :, 14], (0.4472136 , 1.34164079), tol)

                # ncdump -f c -v target_mean 20200301T03Z.nc | grep "target_mean(0,2,3)"
                expected_target = 0.5384477

                # ncdump -f c -v predictors 20200301T03Z.nc | grep "predictors(0,2,3,2)"
                expected_target -= 0.5564111
                self.assertAlmostEqual(t[0, 0, 0, 1, 0], expected_target, tol)
                # self.assertAlmostEqual(p, loader[0][0])
            count += 1

        # loader[0]
        self.assertEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
