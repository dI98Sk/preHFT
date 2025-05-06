import asyncio
import websockets
import json
import pandas as pd

async def fetch_bybit_trades():
    uri = "wss://stream.bybit.com/v5/public/linear"

    # Инициализация CSV
    trades_df = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'side'])
    trades_df.to_csv('bybit_trades.csv', index=False)

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                await websocket.send(json.dumps({
                    "op": "subscribe",
                    "args": ["publicTrade.BTCUSDT"]
                }))
                print("✅ Connected to Bybit Trade WebSocket")

                while True:
                    try:
                        data = json.loads(await websocket.recv())

                        if 'topic' in data and data['topic'] == "publicTrade.BTCUSDT":
                            for trade in data.get('data', []):
                                trade_df = pd.DataFrame({
                                    'timestamp': [trade['T']],
                                    'price': [float(trade['p'])],
                                    'volume': [float(trade['v'])],
                                    'side': [trade['S']]
                                })
                                trade_df.to_csv('bybit_trades.csv', mode='a', header=False, index=False)
                                print(f"[TRADE] {trade['S']} {trade['v']} @ {trade['p']}")

                    except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as e:
                        print(f"🔁 Connection error: {e}. Reconnecting...")
                        break
                    except Exception as e:
                        print(f"⚠️ Error: {e}")
                        continue

        except Exception as e:
            print(f"❌ Failed to connect: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

asyncio.run(fetch_bybit_trades())
