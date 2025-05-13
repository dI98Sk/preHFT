import asyncio
import websockets
import json
import pandas as pd
import ssl
import certifi

# Настраиваем SSL для проверки сертификатов
ssl_context = ssl.create_default_context(cafile=certifi.where())

async def fetch_bybit_data():
    uri = "wss://stream.bybit.com/v5/public/linear"

    # Инициализируем CSV
    df = pd.DataFrame(columns=['timestamp', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume'])
    df.to_csv('bybit_data_sol.csv', index=False)

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10, ssl=ssl_context) as websocket:
                await websocket.send(json.dumps({
                    "op": "subscribe",
                    "args": ["orderbook.1.SOLUSDT"]
                }))
                print("Connected to Bybit WebSocket for SOLUSDT")

                while True:
                    try:
                        data = json.loads(await websocket.recv())
                        if 'data' in data and 'b' in data['data'] and 'a' in data['data']:
                            timestamp = data['ts']
                            bids = data['data']['b']
                            asks = data['data']['a']
                            if bids and asks:
                                df = pd.DataFrame({
                                    'timestamp': [timestamp],
                                    'bid_price': [float(bids[0][0])],
                                    'bid_volume': [float(bids[0][1])],
                                    'ask_price': [float(asks[0][0])],
                                    'ask_volume': [float(asks[0][1])]
                                })
                                df.to_csv('bybit_data_sol.csv', mode='a', header=False, index=False)
                                print(f"Saved SOLUSDT: {timestamp}")
                    except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as e:
                        print(f"Connection error: {e}. Reconnecting...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

        except Exception as e:
            print(f"Failed to connect: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

# Запуск
asyncio.run(fetch_bybit_data())