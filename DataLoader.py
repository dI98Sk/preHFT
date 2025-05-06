import pandas as pd
import os

class DataLoader:
    def __init__(self, path_project):
        self.path = path_project

    def load_csv(self, filename):
        path = os.path.join(self.path, filename)
        return pd.read_csv(path)

    def load_data(self):
        orderbook = self.load_csv('bybit_data.csv')
        trades = self.load_csv('bybit_trades.csv')
        onchain = self.load_csv('onchain_data.csv')
        return orderbook, trades, onchain

    def preprocess_orderbook(self, orderbook_df):
        orderbook_df['datetime'] = pd.to_datetime(orderbook_df['timestamp'], unit='ms')
        orderbook_df['price'] = (orderbook_df['bid_price'] + orderbook_df['ask_price']) / 2

        stakan = orderbook_df.sort_values('timestamp').reset_index(drop=True)
        stakan['diff_bid_ask_price'] = stakan['bid_price'] - stakan['ask_price']
        stakan['diff_bid_ask_volume'] = stakan['bid_volume'] - stakan['ask_volume']

        stakan = stakan.groupby('datetime').agg(
            price=('price', 'mean'),
            bid_price=('bid_price', 'mean'),
            bid_volume=('bid_volume', 'mean'),
            ask_price=('ask_price', 'mean'),
            ask_volume=('ask_volume', 'mean'),
            diff_bid_ask_price=('diff_bid_ask_price', 'mean'),
            diff_bid_ask_volume=('diff_bid_ask_volume', 'mean')
        ).reset_index()

        # Добавим также "price" — например, как mid-price
        stakan['price'] = (stakan['bid_price'] + stakan['ask_price']) / 2

        return stakan

    def preprocess_trades(self, trades_df):
        # trades_df['timestamp'] = trades_df['timestamp'] // 1000  # миллисекунды → секунды
        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
        return trades_df

    def preprocess_onchain(self, whale_df):
        # whale_df['timestamp'] = whale_df['timestamp'] // 1000  # миллисекунды → секунды
        whale_df['datetime'] = pd.to_datetime(whale_df['timestamp'], unit='ms')
        return whale_df

    def get_prepared_datasets(self):
        orderbook = self.load_csv('bybit_data.csv')
        trades = self.load_csv('bybit_trades.csv')
        whale_tx = self.load_csv('onchain_data.csv')

        main_df = self.preprocess_orderbook(orderbook)
        trades_df = self.preprocess_trades(trades)
        whale_tx_df = self.preprocess_onchain(whale_tx)

        return main_df, trades_df, whale_tx_df