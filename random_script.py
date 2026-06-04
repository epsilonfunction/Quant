
# https://old.reddit.com/r/quant/comments/1tvj714/update_3_months_after_asking_about_lowlatency/opkk95c/

import time, websocket

p = [0.0]

def m(_, x):
  i = x.find('"p":"') + 5
  q, now = float(x[i : x.find('"', i)]), time.time()
  if q != p[0]:
    print('BUY' if q > p[0] else 'SELL', q, 'TTL', round((time.time() - now) * 1000000, 2), 'us')
  p[0] = q

websocket.WebSocketApp('wss://stream.binance.com:9443/ws/btcusdt@trade', on_message=m).run_forever()

# BUY 65956.73 TTL 3.1 us
# SELL 65956.72 TTL 4.05 us
# SELL 65956.71 TTL 1.91 us
# SELL 65956.7 TTL 0.0 us

