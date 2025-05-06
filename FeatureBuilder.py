import pandas as pd
import numpy as np
from tqdm import tqdm

class FeatureBuilder:
    def __init__(self, windows=None, target_shift=300, target_threshold=0.005):
        if windows is None:
            self.windows = [5, 10, 20, 30, 45, 60, 300, 600, 900, 1800, 3600]
        else:
            self.windows = windows
        self.target_shift = target_shift
        self.target_threshold = target_threshold

    @staticmethod
    def ensure_continuous_time(df, time_col='datetime'):
        df = df.set_index(time_col).sort_index()
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1s')
        df = df.reindex(full_index)
        df = df.ffill()
        return df.rename_axis('datetime').reset_index()

    def build_features(self, main_df, trades_df, whale_tx_df):
        print("[1/6] 📅 Обеспечение непрерывности времени...")
        main_df = self.ensure_continuous_time(main_df)
        main_df = main_df.set_index('datetime')

        print("[2/6] 🔁 Вычисление лог-доходностей и z-скоров по цене...")
        for w in tqdm(self.windows, desc="→ Лог-доходности и Z-Score"):
            main_df[f'log_return_{w}s'] = np.log(main_df['price'] / main_df['price'].shift(w))
            main_df[f'price_zscore_{w}s'] = (main_df['price'] - main_df['price'].rolling(f'{w}s').mean()) / (main_df['price'].rolling(f'{w}s').std() + 1e-6)

        main_df['volume_imbalance'] = (main_df['bid_volume'] - main_df['ask_volume']) / (main_df['bid_volume'] + main_df['ask_volume'] + 1e-6)

        print("[3/6] 📈 Z-скоры спреда и объёмного дисбаланса...")
        for w in tqdm(self.windows, desc="→ Spread & Volume Imbalance Z-Score"):
            window = f'{w}s'
            main_df[f'vol_imbalance_zscore_{w}s'] = (main_df['volume_imbalance'] - main_df['volume_imbalance'].rolling(window).mean()) / (main_df['volume_imbalance'].rolling(window).std() + 1e-6)
            main_df[f'spread_zscore_{w}s'] = (main_df['diff_bid_ask_price'] - main_df['diff_bid_ask_price'].rolling(window).mean()) / (main_df['diff_bid_ask_price'].rolling(window).std() + 1e-6)

        print("[4/6] 🧮 Агрегация трейдов...")
        trades_df['buy_volume'] = np.where(trades_df['side'] == 'Buy', trades_df['volume'], 0)
        trades_df['sell_volume'] = np.where(trades_df['side'] == 'Sell', trades_df['volume'], 0)

        trade_agg = trades_df.groupby('datetime').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'volume': 'sum'
        }).rename_axis('datetime')

        trade_agg = self.ensure_continuous_time(trade_agg.reset_index(), 'datetime').set_index('datetime')
        trade_agg['volume_imbalance'] = (trade_agg['buy_volume'] - trade_agg['sell_volume']) / (trade_agg['buy_volume'] + trade_agg['sell_volume'] + 1e-6)

        for w in tqdm(self.windows, desc="→ Фичи из трейдов"):
            window = f'{w}s'
            trade_agg[f'volume_zscore_{w}s'] = (trade_agg['volume'] - trade_agg['volume'].rolling(window).mean()) / (trade_agg['volume'].rolling(window).std() + 1e-6)
            trade_agg[f'imbalance_zscore_{w}s'] = (trade_agg['volume_imbalance'] - trade_agg['volume_imbalance'].rolling(window).mean()) / (trade_agg['volume_imbalance'].rolling(window).std() + 1e-6)
            trade_agg[f'trade_trend_{w}s'] = trade_agg['volume'].rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

        print("[5/6] 🐋 Обработка whale-транзакций...")
        whale_tx_df = whale_tx_df.set_index('datetime').sort_index()
        whale_min = whale_tx_df.resample('1min').mean()
        whale_sec = whale_min.resample('1s').interpolate(method='time')
        whale_sec = self.ensure_continuous_time(whale_sec.reset_index(), 'datetime').set_index('datetime')

        for w in tqdm(self.windows, desc="→ Фичи из whale_tx"):
            window = f'{w}s'
            whale_sec[f'whale_tx_zscore_{w}s'] = (whale_sec['whale_tx'] - whale_sec['whale_tx'].rolling(window).mean()) / (whale_sec['whale_tx'].rolling(window).std() + 1e-6)
            whale_sec[f'whale_tx_trend_{w}s'] = whale_sec['whale_tx'].rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
            median_val = whale_sec["whale_tx"].rolling(w).median()
            whale_sec[f"whale_tx_spike_{w}s"] = (whale_sec["whale_tx"] > 5 * median_val).astype(int)

        print("[6/6] 🔗 Объединение всех фичей...")
        df = main_df.join(trade_agg, how='left', rsuffix='_trade')
        df = df.join(whale_sec, how='left', rsuffix='_whale')
        df = df.fillna(0).reset_index()

        return df

    def add_target(self, dataset):
        # создаём целевую колонку (target_...) и её имя формируется динамически:
        # 	•	self.target_shift = 300
        # 	•	self.target_threshold = 0.005
        # 	•	Значит, имя колонки будет: target_300_50bp - Важно для файла ModelTrainer
        print("[+] 🎯 Генерация целевой переменной...")
        min_shift = 0
        max_shift = self.target_shift

        print(f"→ 📊 Сдвиги цен вперёд до {max_shift} сек...")
        future_prices = pd.DataFrame({
            f'shift_{i}': dataset['price'].shift(-i) for i in range(min_shift, max_shift + 1)
        })

        print("→ 📈 Расчёт максимумов/минимумов будущих цен...")
        max_future_price = future_prices.max(axis=1)
        min_future_price = future_prices.min(axis=1)

        price_now = dataset['price']
        price_up_pct = (max_future_price - price_now) / price_now
        price_down_pct = (min_future_price - price_now) / price_now

        print(f"→ 🏷️ Присвоение классов на основе порога {self.target_threshold:.4f}...")

        def get_label(up, down):
            if up > self.target_threshold and down < -self.target_threshold:
                return 0
            elif up > self.target_threshold:
                return 1
            elif down < -self.target_threshold:
                return 2
            else:
                return 0

        labels = []
        for u, d in tqdm(zip(price_up_pct, price_down_pct), total=len(price_up_pct), desc="→ Классификация"):
            if not np.isnan(u) and not np.isnan(d):
                labels.append(get_label(u, d))
            else:
                labels.append(np.nan)

        dataset[f'target_{max_shift}_{int(self.target_threshold * 10000)}bp'] = labels


        # Legasy часть, так было до tqdm
        # dataset[f'target_{max_shift}_{int(self.target_threshold*10000)}bp'] = [
        #     get_label(u, d) if not np.isnan(u) and not np.isnan(d) else np.nan
        #     for u, d in zip(price_up_pct, price_down_pct)
        # ]

        return dataset

    def build_and_save(self, main_df, trades_df, whale_tx_df, output_path):
        df = self.build_features(main_df, trades_df, whale_tx_df)
        df = self.add_target(df)

        drop_cols = ['bid_price', 'ask_price', 'diff_bid_ask_volume', 'price']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        df.to_parquet(output_path, index=False)
        print(f"[+] Dataset saved to: {output_path}")

        return df
