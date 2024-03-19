# import psutil
# import os
# import datetime
# import time
# import threading

# def track_time():
#     while True:
#         now = datetime.datetime.now()
#         milliseconds = now.microsecond // 1000

#         # 현재 프로세스의 PID를 가져옵니다.
#         pid = os.getpid()
#         p = psutil.Process(pid)
#         memory_usage = p.memory_info().rss / (1024 * 1024)  # 메모리 사용량을 메가바이트 단위로 변환합니다.

#         print(f"Time: {now} : {milliseconds}ms, Memory usage: {memory_usage} MB")
#         time.sleep(1)  # 1ms 대기

# # 별도의 스레드에서 시간 추적 함수를 실행합니다.
# threading.Thread(target=track_time).start()

import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import pytz
import ccxt

exchange = ccxt.binance({
    'apiKey': 'txuapw1jXplRdwUiLDIFYNMlSCo4WHDklahUjGsHESYQMUKhg7WC80EW6L3newBL',
    'secret': 'siFn8qN0z56V4iNWfCnwydWOGEAdwa75FLEqNTVeSduF7XSNG2IfhusdRN83YP50',
    'options': {
        'defaultType': 'future'
    }
})

while True:
    a = dt.datetime.fromtimestamp(time.time()).second
    if a == 2:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=2)
        ohlcv[0][0] = dt.datetime.fromtimestamp(ohlcv[0][0] / 1000)
        ohlcv[0][0] = ohlcv[0][0].strftime("%Y-%m-%d %H:%M:%S")
        price = ohlcv[0]  # 1분 전의 시가
        print(price)
        time.sleep(1)
