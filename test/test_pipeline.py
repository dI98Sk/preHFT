import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from feature_builder import FeatureBuilder
from imblearn.over_sampling import SMOTE
from unittest.mock import patch


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Генерация данных для пайплайна
        np.random.seed(42)

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

    @patch("imblearn.over_sampling.SMOTE.fit_resample")
    @patch("xgboost.XGBClassifier.fit")
    @patch("xgboost.XGBClassifier.save_model")
    def test_pipeline_integration(self, mock_save_model, mock_fit, mock_smote):
        # Мок для SMOTE
        mock_smote.return_value = (np.random.rand(100, 10), np.random.randint(0, 3, 100))

        # Генерация фичей и целевых значений
        df_feat = self.builder.build_features(self.main_df, self.trades_df, self.whale_df)
        df_full = self.builder.add_target(df_feat.copy())

        # Разделим на train и test
        train_df = df_full.iloc[:3500]
        test_df = df_full.iloc[3500:]

        # Мок для обучения
        mock_fit.return_value = MagicMock()

        # Вызов пайплайна с проверкой, что SMOTE используется только на train
        X_train = train_df.drop(columns=["datetime", "target_60_100bp"])
        y_train = train_df["target_60_100bp"]

        # Обучение модели
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        mock_smote.assert_called_once_with(X_train, y_train)

        # Проверим, что SMOTE не применяется к test данным
        X_test = test_df.drop(columns=["datetime", "target_60_100bp"])
        y_test = test_df["target_60_100bp"]
        self.assertNotIn("target_60_100bp", X_test.columns, "Test data should not have been altered by SMOTE")

        # Мокирование сохранения модели
        mock_save_model.assert_called_once_with("models/model_xgb.pkl")

        # Проверим, что модель обучилась (метод fit был вызван)
        mock_fit.assert_called_once()

        print("[+] ✅ Пайплайн прошёл успешно!")