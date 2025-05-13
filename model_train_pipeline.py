# from ModelPredictor import ModelPredictor
# from ModelTrainerUpdate import ModelTrainerUpdate
import pandas as pd

from ModelPredictor import ModelPredictor
from ModelTrainer import ModelTrainer
from ModelTrainerUpdate import ModelTrainerUpdate

# Этот файл сформирован в тестовых целях, главный пайплайн слишком тяжелый для постоянного запуска,
# после отлатки блоки будут попадать в основной файл
if __name__ == "__main__":

    # На обновленном паркете работает, результат надо проанализировать
    trainer = ModelTrainer(data_path='dataset_XGB.parquet',
                           model_output_path='models/model_xgb.pkl', target_col='target_300_50bp')
    # Название целевого столбца (должен совпадать с тем, что добавлялся в FeatureBuilder)
    # target_col = 'target_300_50bp' например, для target_shift=300, threshold=0.005
    trainer.run()


    # updateTrainer = ModelTrainerUpdate(data_path='/Users/papaskakun/PycharmProjects/preHFT/dataset_XGB.parquet',
    #                                    model_output_path='models/model_xgb.pkl', target_col='target_300_50bp',
    #                                    use_cv=True,  # Для включения кросс-валидации
    #                                    n_splits=5,  # Количество фолдов для кросс-валидации
    #                                    use_smote=True,  # Использовать ли SMOTE для балансировки данных
    #                                    random_state=42
    #                                    # Устанавливаем фиксированное значение random_state для воспроизводимости
    #                                    )
    #
    # updateTrainer.run()





