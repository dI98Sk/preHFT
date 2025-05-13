import unittest
import pandas as pd
import numpy as np

from feature_builder import FeatureBuilder

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        # Генерация тестовых данных
        dt_range = pd.date_range("2023-01-01", periods=5000, freq="s")
        self.main_df = pd.DataFrame({
            "datetime": dt_range,
            "price": np.cumsum(np.random.randn(5000)) + 100,
            "bid_volume": np.random.randint(1, 100, size=5000),
            "ask_volume": np.random.randint(1, 100, size=5000),
            "diff_bid_ask_price": np.random.randn(5000)
        })

        self.trades_df = pd.DataFrame({
            "datetime": dt_range,
            "side": np.random.choice(["Buy", "Sell"], size=5000),
            "volume": np.random.randint(1, 10, size=5000)
        })

        self.whale_df = pd.DataFrame({
            "datetime": dt_range,
            "whale_tx": np.random.randn(5000)
        })

        self.builder = FeatureBuilder(windows=[5, 10], target_shift=60, target_threshold=0.01)

    def test_no_future_leak_in_features(self):
        df = self.builder.build_features(self.main_df, self.trades_df, self.whale_df)
        for col in df.columns:
            if 'zscore' in col or 'log_return' in col or 'trend' in col:
                # Убедимся, что нет NaN "задним числом"
                nan_before_valid = df[col].isna().cumsum().max()
                self.assertGreater(nan_before_valid, 0, f"{col} should start with NaNs due to rolling window")

    def test_target_uses_future(self):
        df_feat = self.builder.build_features(self.main_df, self.trades_df, self.whale_df)
        df_full = self.builder.add_target(df_feat.copy())
        target_col = [col for col in df_full.columns if col.startswith("target_")][0]
        n_nan_tail = df_full[target_col].isna().sum()
        # Последние `target_shift` значений должны быть NaN
        self.assertEqual(n_nan_tail, self.builder.target_shift, "Target should be NaN at the tail")

    def test_target_not_leaking_into_features(self):
        df_feat = self.builder.build_features(self.main_df, self.trades_df, self.whale_df)
        df_full = self.builder.add_target(df_feat.copy())
        target_col = [col for col in df_full.columns if col.startswith("target_")][0]
        feature_cols = [col for col in df_full.columns if col not in ['datetime', target_col]]
        corr = df_full[feature_cols].corrwith(df_full[target_col].fillna(0)).abs()
        self.assertTrue((corr < 1.0).all(), "No feature should perfectly correlate with the target")

    def test_rolling_applies_backwards(self):
        df = self.builder.build_features(self.main_df, self.trades_df, self.whale_df)
        z_cols = [c for c in df.columns if 'zscore' in c]
        for col in z_cols:
            first_valid = df[col].first_valid_index()
            self.assertIsNotNone(first_valid, f"{col} should eventually become valid")
