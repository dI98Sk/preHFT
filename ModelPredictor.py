import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelPredictor:
    def __init__(self, model_path, data_path, target_col):
        self.model_path = model_path
        self.data_path = data_path
        self.target_col = target_col
        self.model = None

    def load_model(self):
        # Загрузка модели: В методе load_model загружается ранее сохраненная модель с помощью joblib.load().
        # Загрузка сохраненной модели
        print(f"[+] Загрузка модели из {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print("[+] Модель загружена.")

    def load_data(self):
        # Загрузка новых данных
        print("[+] Загрузка данных для предсказаний...")
        df = pd.read_parquet(self.data_path)
        print(f"[+] Данные загружены: {df.shape}")
        return df

    def preprocess(self, df):
        # В методе preprocess мы выполняем те же шаги, что и при обучении модели:
        # удаляем ненужные столбцы и разделяем данные на признаки и целевую переменную.
        print("[+] Предобработка данных...")

        # Удаляем 'datetime', если есть
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])

        # Отделяем целевую переменную
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(int)

        print("[+] Target distribution:")
        print(y.value_counts(normalize=True))

        return X, y

    def predict(self, X):
        # использует загруженную модель для получения предсказаний на новых данных
        # Получаем предсказания от модели
        print("[+] Предсказания...")
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate(self, y_true, y_pred):
        # Оценка результатов предсказания
        print("[+] 📊 Оценка модели...")

        print("[+] Classification Report:")
        print(classification_report(y_true, y_pred, digits=2))

        print("[+] Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def save_predictions(self, y_pred):
        # Сохранение предсказаний в CSV
        os.makedirs('predictions', exist_ok=True)
        predictions_df = pd.DataFrame(y_pred, columns=['prediction'])
        predictions_df.to_csv('predictions/predictions.csv', index=False)
        print("[+] Предсказания сохранены в predictions/predictions.csv")

    def run(self):
        # Загружаем модель
        self.load_model()

        # Загружаем данные
        df = self.load_data()

        # Предобработка данных
        X, y = self.preprocess(df)

        # Получаем предсказания
        y_pred = self.predict(X)

        # Оценка модели
        self.evaluate(y, y_pred)

        # Сохранение предсказаний
        self.save_predictions(y_pred)


