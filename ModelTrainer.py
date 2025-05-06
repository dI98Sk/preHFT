from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import os
from tqdm import tqdm
import seaborn as sns  # Для тепловой карты

class ModelTrainer:
    def __init__(self, data_path, model_output_path, target_col):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.target_col = target_col

    def load_data(self):
        print("[+] Загрузка данных...")
        df = pd.read_parquet(self.data_path)
        print(f"[+] Data loaded: {df.shape}")
        return df

    def preprocess(self, df, target_col):
        print("[+] Предобработка данных...")

        # Удаляем 'datetime', если есть
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])

        # Отделяем целевую переменную
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)  # убедимся, что классы — это int

        # Разделение на тренировочные и тестовые данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[+] Target distribution in training set:")
        print(y_train.value_counts(normalize=True))

        print("[+] Target distribution in test set:")
        print(y_test.value_counts(normalize=True))

        return X_train, X_test, y_train, y_test

    def evaluate(self, model, X_test, y_test):
        print("[+] 📊 Оценка модели...")
        y_pred = model.predict(X_test)

        print("[+] Classification Report:")
        print(classification_report(y_test, y_pred, digits=2))

        print("[+] Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_and_save_target_distribution(self, y_train, y_test):
        # Создаем директорию для хранения графиков, если она еще не существует
        os.makedirs('plots', exist_ok=True)

        # График распределения для y_train
        plt.figure(figsize=(6, 4))
        y_train.value_counts(normalize=True).sort_index().plot(kind='bar', title='Target distribution (train)')
        plt.ylabel('Proportion')
        plt.xlabel('Target class')
        plt.tight_layout()
        plt.savefig('plots/target_distribution_train.png')
        plt.close()

        # График распределения для y_test
        plt.figure(figsize=(6, 4))
        y_test.value_counts(normalize=True).sort_index().plot(kind='bar', title='Target distribution (test)')
        plt.ylabel('Proportion')
        plt.xlabel('Target class')
        plt.tight_layout()
        plt.savefig('plots/target_distribution_test.png')
        plt.close()

        print("[+] Target distribution saved as PNG.")

    def train(self, X_train, y_train, class_weights):
        # Применяем SMOTE для увеличения примеров для классов 1 и 2
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Используем XGBoost без параметров use_label_encoder и scale_pos_weight
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            eval_metric='mlogloss',  # Убедитесь, что метрика указана, если хотите отслеживать прогресс
        )

        # Оборачивание обучения с tqdm для отображения прогресса
        eval_set = [(X_train_resampled, y_train_resampled)]  # Данные для отслеживания прогресса
        model.fit(X_train_resampled, y_train_resampled, eval_set=eval_set, verbose=False)

        return model

    def save_model(self, model):
        joblib.dump(model, self.model_output_path)
        print(f"[+] Model saved to: {self.model_output_path}")

    def run(self):
        # Загрузка данных
        print("[+] Загрузка данных...")
        df = self.load_data()

        # Предобработка
        print("[+] Предобработка данных...")
        X_train, X_test, y_train, y_test = self.preprocess(df, self.target_col)

        # Вычисляем веса классов
        print("[+] ⚖️ Вычисление class_weights...")
        '''
        Распределение целевой переменной в тренировочном наборе:
        	•	Класс 0 (основной): 98.59% данных.
        	•	Класс 1: 0.70% данных.
        	•	Класс 2: 0.70% данных.

        Распределение целевой переменной в тестовом наборе:
        	•	Класс 0: 98.58% данных.
        	•	Класс 1: 0.75% данных.
        	•	Класс 2: 0.67% данных.
        	
        Вычисление весов классов:
	        •	Вес для класса 0: 0.34
	        •	Вес для класса 1: 47.37
	        •	Вес для класса 2: 47.40
	        •	Эти веса были вычислены, чтобы помочь модели лучше работать с дисбалансом классов, 
	            компенсируя меньшую представленность классов 1 и 2.  	
        '''
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"[+] Class weights: {class_weights}")

        # Визуализация распределения классов и сохранение в файл
        self.plot_and_save_target_distribution(y_train, y_test)

        # Обучение модели
        print("[+] 🧠 Обучение модели XGBoost...")
        model = self.train(X_train, y_train, class_weights)

        # Оценка модели
        print("[+] 📊 Оценка модели...")
        '''
        Оценка модели:
	•	Classification Report:
	•	Класс 0 (основной):
	•	Precision (точность): 1.00 — модель верно классифицирует все примеры класса 0.
	•	Recall (полнота): 1.00 — модель правильно классифицирует все примеры класса 0.
	•	F1-score: 1.00 — идеальный результат для класса 0.
	•	Класс 1:
	•	Precision: 0.92 — модель верно классифицирует 92% примеров класса 1.
	•	Recall: 1.00 — модель правильно классифицирует все примеры класса 1.
	•	F1-score: 0.96 — довольно высокое значение для класса 1.
	•	Класс 2:
	•	Precision: 0.91 — модель верно классифицирует 91% примеров класса 2.
	•	Recall: 0.99 — модель правильно классифицирует 99% примеров класса 2.
	•	F1-score: 0.95 — высокое значение для класса 2.
	•	Общие метрики:
	•	Accuracy (точность): 1.00 — модель верно классифицирует все примеры (включая редкие классы).
	•	Macro average (среднее по классам): 0.94 (для precision), 1.00 (для recall), 0.97 (для f1-score)
	      — средние значения по всем классам. Отличный результат, учитывая дисбаланс классов.
	•	Weighted average (взвешенное среднее): 1.00 — средние значения с учетом веса классов,
	     что подтверждает высокую точность и полноту модели.
        '''
        self.evaluate(model, X_test, y_test)

        # Сохранение модели
        print(f"[+] Model saved to: {self.model_output_path}")
        self.save_model(model)


