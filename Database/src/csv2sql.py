import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os

CSV_FILE = "/app/hourly_hype_with_indicators.csv"

#############################################
# 1. 캔들 데이터 불러오기
#############################################
def candle_chart(interval):
    current_time = int(pd.Timestamp.now().timestamp() * 1000)
    api_url = "https://api.hyperliquid.xyz/info"

    body_candles = {
        "type": "candleSnapshot",
        "req": {
            "coin": "HYPE",
            "interval": interval,
            "startTime": 0,
            "endTime": current_time
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=body_candles)

    if response.status_code == 200:
        print("API 요청 성공, 데이터 파싱 중...")
        candles = response.json()
        df = pd.DataFrame(candles)

        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close",
                           "v": "Volume", "n": "total_transaction"}, inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        df.drop(columns=["T", "s", "i"], errors="ignore", inplace=True)
        return df
    else:
        print(f"API 요청 실패! 상태 코드: {response.status_code}")
        return None


#############################################
# 2. 볼린저 밴드
#############################################
def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    return pd.DataFrame({
        'SMA': rolling_mean,
        'Upper_Band': rolling_mean + (rolling_std * num_std),
        'Lower_Band': rolling_mean - (rolling_std * num_std)
    })


#############################################
# 3. MACD, RSI, EMA
#############################################
def compute_technical_indicators(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.DataFrame({
        'MACD': macd,
        'MACD_Signal': signal,
        'RSI': rsi,
        'EMA_50': df['Close'].ewm(span=50, adjust=False).mean(),
        'EMA_200': df['Close'].ewm(span=200, adjust=False).mean()
    })


#############################################
# 4. CSV 저장
#############################################
def save_to_csv(df):
    df.tail(1).to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=True)
    print("CSV 최신 데이터 1줄 저장 완료")


#############################################
# 5. CSV → MySQL UPSERT
#############################################
def csv_to_sql():
    if not os.path.exists(CSV_FILE):
        print("CSV 파일이 존재하지 않아 SQL 업데이트 불가")
        return

    df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    engine = create_engine("mysql+pymysql://hype_hourly_user:1234@mysql:3306/hype_hourly_db")

    with engine.begin() as conn:
        for _, row in df.reset_index().iterrows():
            query = text("""
                INSERT INTO hourly_hype (t, Open, Close, High, Low, Volume, total_transaction,
                                         SMA, Upper_Band, Lower_Band, MACD, MACD_Signal, RSI, EMA_50, EMA_200)
                VALUES (:t, :Open, :Close, :High, :Low, :Volume, :total_transaction,
                        :SMA, :Upper_Band, :Lower_Band, :MACD, :MACD_Signal, :RSI, :EMA_50, :EMA_200)
                ON DUPLICATE KEY UPDATE
                    Open=VALUES(Open),
                    Close=VALUES(Close),
                    High=VALUES(High),
                    Low=VALUES(Low),
                    Volume=VALUES(Volume),
                    total_transaction=VALUES(total_transaction),
                    SMA=VALUES(SMA),
                    Upper_Band=VALUES(Upper_Band),
                    Lower_Band=VALUES(Lower_Band),
                    MACD=VALUES(MACD),
                    MACD_Signal=VALUES(MACD_Signal),
                    RSI=VALUES(RSI),
                    EMA_50=VALUES(EMA_50),
                    EMA_200=VALUES(EMA_200)
            """)
            conn.execute(query, row.to_dict())
    print("CSV → MySQL 동기화 완료")


#############################################
# 6. 메인 실행
#############################################
if __name__ == "__main__":
    df = candle_chart(interval="1h")
    if df is not None:
        bb_df = compute_bollinger_bands(df["Close"])
        tech_df = compute_technical_indicators(df)
        merged_df = pd.concat([df, bb_df, tech_df], axis=1).dropna()

        save_to_csv(merged_df)
        csv_to_sql()
    else:
        print("데이터 없음, 종료")
