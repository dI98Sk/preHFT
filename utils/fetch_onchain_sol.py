import asyncio
import time
import pandas as pd
from solana.rpc.async_api import AsyncClient

# Подключаемся к Solana
client = AsyncClient("https://api.mainnet-beta.solana.com")

# Whale threshold
WHALE_THRESHOLD = 10 * 10**9  # Порог в lamports (1 SOL = 10^9 lamports)

# Инициализируем CSV (если не существует)
CSV_FILE = 'onchain_data_solana.csv'
try:
    pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df_init = pd.DataFrame(columns=['timestamp', 'whale_tx'])
    df_init.to_csv(CSV_FILE, index=False)

async def fetch_onchain_data():
    try:
        # Получаем текущий слот
        last_slot_resp = await client.get_slot()
        print("Ответ на get_slot():", last_slot_resp)

        if hasattr(last_slot_resp, 'value'):
            last_slot = last_slot_resp.value  # Извлекаем значение слота
            print(f"Текущий слот: {last_slot}")
        else:
            print(f"Ошибка: ответ не содержит 'value'. Ответ: {last_slot_resp}")
            return

    except Exception as e:
        print(f"Ошибка при получении слота: {e}")
        return

    while True:
        try:
            # Получаем новый слот
            current_slot_resp = await client.get_slot()
            print("Ответ на get_slot():", current_slot_resp)

            if hasattr(current_slot_resp, 'value'):
                current_slot = current_slot_resp.value  # Извлекаем значение слота
                print(f"Текущий слот: {current_slot}")
            else:
                print(f"Ошибка: ответ не содержит 'value'. Ответ: {current_slot_resp}")
                return

            whale_tx_count = 0

            # Получаем блоки между last_slot и current_slot
            for slot in range(last_slot + 1, current_slot + 1):
                # Получаем недавний блок
                blockhash_resp = await client.get_latest_blockhash()
                print("Ответ на get_recent_blockhash():", blockhash_resp)

                if 'result' in blockhash_resp and blockhash_resp['result']:
                    blockhash = blockhash_resp['result']['value']['blockhash']
                    print(f"Получен блок с хешем: {blockhash}")
                else:
                    print(f"Ошибка: не удалось получить blockhash. Ответ: {blockhash_resp}")

            timestamp = int(time.time())
            df = pd.DataFrame({'timestamp': [timestamp], 'whale_tx': [whale_tx_count]})
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Slots {last_slot+1} to {current_slot} → Whale TX: {whale_tx_count}")

            last_slot = current_slot  # Обновляем last_slot для следующей итерации

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(fetch_onchain_data())