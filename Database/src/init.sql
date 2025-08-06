CREATE TABLE IF NOT EXISTS hourly_hype (
    t DATETIME PRIMARY KEY,
    Open FLOAT,
    Close FLOAT,
    High FLOAT,
    Low FLOAT,
    Volume FLOAT,
    total_transaction INT,
    SMA FLOAT,
    Upper_Band FLOAT,
    Lower_Band FLOAT,
    MACD FLOAT,
    MACD_Signal FLOAT,
    RSI FLOAT,
    EMA_50 FLOAT,
    EMA_200 FLOAT
);
