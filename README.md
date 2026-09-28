# Signal Forge — Web Crypto Signal Engine

Ứng dụng Flask/PWA phân tích crypto bằng CCXT + pandas + pandas-ta-classic + GradientBoosting + RandomForest meta-labeling.

## Lưu ý quan trọng

- GitHub dùng để lưu source code; GitHub Pages **không chạy được Flask/Python backend**.
- Có thể kết nối repository GitHub này với Render hoặc một VPS để chạy web.
- Hệ thống chỉ hỗ trợ phân tích/quyết định; **không đảm bảo lợi nhuận hay độ chính xác 100%**.
- Dữ liệu OHLCV lấy từ sàn qua CCXT. API key giao dịch không được yêu cầu để đọc OHLCV công khai.
- Model và `signals.json` được lưu trên filesystem của instance. Với hosting không có persistent disk, dữ liệu/model có thể mất khi redeploy/restart.

## 6 lớp bắt buộc

1. Rule-based: ít nhất 3/4 RSI, Bollinger Band, Stochastic, MACD cùng hướng.
2. HTF: hướng giá khung lớn so với EMA50 phải cùng hướng.
3. Regime: ADX >= 22 và Choppiness <= 55.
4. GradientBoosting: confidence >= 70% và cùng hướng rule.
5. RandomForest meta-label: probability >= 70%.
6. Walk-forward TimeSeriesSplit accuracy >= 62%.

Composite confidence:
`0.30*rule + 0.35*ML + 0.20*meta + 0.15*walk-forward`

## Deploy bằng GitHub + Render

1. Tạo repository mới trên GitHub.
2. Upload toàn bộ file trong thư mục này.
3. Trên Render chọn **New Web Service** → kết nối repository.
4. Build command:
   `pip install -r requirements.txt`
5. Start command:
   `gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT app:app`
6. Runtime Python 3.11.
7. Nếu dùng Telegram, thêm Environment Variables:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`

Sau khi deploy, mở URL Render cấp.

## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Mở `http://127.0.0.1:5000`.

## Train

Mở Settings → Save Config → Train Model. Sau khi train thành công mới dùng Scan Now/Scanner.
