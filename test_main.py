import unittest
from unittest.mock import patch
import pandas as pd
from main import load_data
from main import verify_data_shapes, preprocess_data
import numpy as np
from unittest.mock import MagicMock


class TestMelanomaClassification(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        # Mocking pandas.read_csv to return a DataFrame
        mock_read_csv.return_value = pd.DataFrame()

        train_df, test_df = load_data('valid_train_path.csv', 'valid_test_path.csv')

        # Assert that the returned objects are indeed DataFrames
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)

    @patch('pandas.read_csv')
    def test_load_data_failure(self, mock_read_csv):
        # Mocking pandas.read_csv to raise FileNotFoundError
        mock_read_csv.side_effect = FileNotFoundError

        with self.assertRaises(FileNotFoundError):
            load_data('invalid_train_path.csv', 'invalid_test_path.csv')


class TestVerifyDataShapes(unittest.TestCase):
    @patch('main.load_data')
    @patch('main.preprocess_data')
    @patch('builtins.print')
    def test_verify_data_shapes(self, mocked_print, mocked_preprocess_data, mocked_load_data):
        # Setup mock return values to simulate the data shapes
        mocked_load_data.return_value = (MagicMock(), MagicMock())
        mocked_preprocess_data.return_value = (MagicMock(shape=(100, 5)), MagicMock())

        # Call the function under test
        verify_data_shapes('dummy_train_path.csv', 'dummy_test_path.csv')

        # Assert print was called with the expected string
        mocked_print.assert_any_call('X_train shape: (100, 5)')


class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        df = pd.DataFrame({
            'age_approx': [30, np.nan, 45],
            'sex': ['male', 'female', np.nan],
            'anatom_site_general_challenge': ['torso', 'lower extremity', np.nan],
            'target': [0, 1, 0]
        })
        X, y = preprocess_data(df, is_train=True)
        self.assertEqual(X.shape, (3, 6))
        self.assertTrue('age_group' in X.columns)
        self.assertFalse(np.any(pd.isnull(X)))
        self.assertEqual(len(y), 3)


if __name__ == '__main__':
    unittest.main()
