# Инициализируем загрузчик данных
from DataLoader import DataLoader
from FeatureBuilder import FeatureBuilder

dl = DataLoader(path_project='/Users/papaskakun/PycharmProjects/preHFT/data')

# Получаем подготовленные данные
main_df, trades_df, whale_tx_df = dl.get_prepared_datasets()

# Инициализируем билдера фичей и таргетов
fb = FeatureBuilder(target_shift=300, target_threshold=0.005)

# Обработка + сохранение
df_ready = fb.build_and_save(
    main_df=main_df,
    trades_df=trades_df,
    whale_tx_df=whale_tx_df,
    output_path='/Users/papaskakun/PycharmProjects/preHFT/dataset_XGB.parquet'
)