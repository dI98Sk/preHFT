import os
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm  # Добавлен импорт tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class ModelTrainerUpdate:
    '''
    Этот класс был создан как более суровая версия тренировки, с разными условиями проверки для предотвращения утечек
    и подсматривания в будущее
    '''
    def __init__(self, data_path, model_output_path, target_col,
                 use_cv=True, n_splits=5, use_smote=True, random_state=42,
                 model_params=None):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.target_col = target_col
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.use_smote = use_smote
        self.random_state = random_state
        self.model_params = model_params or {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': random_state
        }

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")

        logging.info("Загрузка данных...")
        df = pd.read_parquet(self.data_path)

        logging.info("Проверка NaN до ffill():\n%s", df.isna().sum())
        df = df.ffill()
        logging.info("Проверка NaN после ffill():\n%s", df.isna().sum())

        logging.info("Загружено данных: %s", df.shape)
        return df

    def preprocess(self, df):
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])

        # Дополнительная обработка выбросов (например, используя IQR)
        for col in df.select_dtypes(include=['float64', 'int64']):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        logging.info("Target distribution (train):\n%s", y_train.value_counts(normalize=True))
        logging.info("Target distribution (test):\n%s", y_test.value_counts(normalize=True))

        # Стандартизация признаков
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def apply_smote(self, X, y):
        logging.info("Применение SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        return smote.fit_resample(X, y)

    def train(self, X_train, y_train, class_weights, scale_pos_weight=None):
        if self.use_cv:
            logging.info("Кросс-валидация (%d-fold)", self.n_splits)
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            best_model = None
            best_score = -1.0

            for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X_train, y_train)), total=self.n_splits, desc="Кросс-валидация"):
                logging.info("[*] Fold %d", fold + 1)

                X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
                X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

                if self.use_smote:
                    X_fold_train, y_fold_train = self.apply_smote(X_fold_train, y_fold_train)

                model = XGBClassifier(
                    **self.model_params,
                    num_class=len(np.unique(y_fold_train)),
                    scale_pos_weight=scale_pos_weight
                )
                model.fit(X_fold_train, y_fold_train,
                          eval_set=[(X_fold_val, y_fold_val)],
                          verbose=False)

                y_val_pred = model.predict(X_fold_val)
                f1 = f1_score(y_fold_val, y_val_pred, average='macro')
                logging.info("Fold %d - F1 macro: %.4f", fold + 1, f1)

                if f1 > best_score:
                    logging.info("🔝 Новая лучшая модель (F1 macro: %.4f)", f1)
                    best_model = model
                    best_score = f1

            return best_model

        else:
            if self.use_smote:
                X_train, y_train = self.apply_smote(X_train, y_train)

            logging.info("Обучение модели без кросс-валидации...")
            model = XGBClassifier(
                **self.model_params,
                num_class=len(np.unique(y_train)),
                scale_pos_weight=scale_pos_weight
            )
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
            return model

    def evaluate(self, model, X_test, y_test):
        logging.info("Оценка модели...")
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, digits=2, output_dict=True)
        logging.info("%s", classification_report(y_test, y_pred, digits=2))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('reports', exist_ok=True)
        with open(f'reports/classification_report_{timestamp}.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred, digits=2))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f'reports/confusion_matrix_{timestamp}.png')
        plt.close()

    def plot_and_save_target_distribution(self, y_train, y_test):
        os.makedirs('plots', exist_ok=True)
        for y, name in [(y_train, "train"), (y_test, "test")]:
            plt.figure(figsize=(6, 4))
            y.value_counts(normalize=True).sort_index().plot(kind='bar', title=f'Target distribution ({name})')
            plt.ylabel('Proportion')
            plt.xlabel('Target class')
            plt.tight_layout()
            plt.savefig(f'plots/target_distribution_{name}.png')
            plt.close()

    def save_model(self, model):
        joblib.dump(model, self.model_output_path)
        logging.info("Модель сохранена: %s", self.model_output_path)

    def run(self):
        logging.info("🚀 Запуск pipeline...")
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess(df)

        logging.info("Вычисление весов классов...")
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        logging.info("Class weights: %s", class_weights)

        scale_pos_weight = None
        if len(classes) == 2:
            scale_pos_weight = (y_train == classes[0]).sum() / max((y_train == classes[1]).sum(), 1)
            logging.info("scale_pos_weight: %.2f", scale_pos_weight)

        self.plot_and_save_target_distribution(y_train, y_test)
        logging.info("🔧 Обучение модели...")
        model = self.train(X_train, y_train, class_weights, scale_pos_weight)
        self.evaluate(model, X_test, y_test)
        self.save_model(model)
        logging.info("✅ Готово!")