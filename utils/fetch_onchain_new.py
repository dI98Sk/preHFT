import asyncio
import time
import pandas as pd
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Подключаемся к BSC
# Через Web3.HTTPProvider к публичному BSC-ноду:
w3 = Web3(Web3.HTTPProvider('https://bsc-dataseed1.binance.org'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Whale threshold
WHALE_THRESHOLD = 10  # в ETH/BNB

# Инициализируем CSV (если не существует)
CSV_FILE = 'onchain_data.csv'
try:
    pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df_init = pd.DataFrame(columns=['timestamp', 'whale_tx'])
    df_init.to_csv(CSV_FILE, index=False)

async def fetch_onchain_data():
    last_block = w3.eth.block_number

    while True:
        try:
            current_block = w3.eth.block_number
            whale_tx_count = 0

            for block_number in range(last_block + 1, current_block + 1):
                block = w3.eth.get_block(block_number, full_transactions=True)
                for tx in block['transactions']:
                    value_eth = w3.fromWei(tx['value'], 'ether')
                    if value_eth > WHALE_THRESHOLD:
                        whale_tx_count += 1

            timestamp = int(time.time())
            df = pd.DataFrame({'timestamp': [timestamp], 'whale_tx': [whale_tx_count]})
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Blocks {last_block+1} to {current_block} → Whale TX: {whale_tx_count}")

            last_block = current_block

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(fetch_onchain_data())