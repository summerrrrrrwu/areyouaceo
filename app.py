from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# 載入模型
model = joblib.load('stacking_model.pkl')

@app.route("/")
def home():
    return "Hello, this is a production WSGI server using Waitress!"

@app.route('/predict', methods=['POST'])
def predict():
    # 獲取請求的輸入數據（示例用隨機數模擬）
    data = request.json.get('features')  # 假設前端發送 JSON 格式的特徵
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # 轉換輸入數據為 numpy array
    input_features = np.array(data).reshape(1, -1)

    # 使用模型預測
    probability = model.predict_proba(input_features)[:, 1][0] * 100

    # 返回預測結果
    return jsonify({'ceo_probability': round(probability, 2)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        # 嘗試使用 Waitress 啟動
        from waitress import serve
        print("Running with Waitress server...")
        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        # 如果未安裝 Waitress，使用 Flask 內建開發伺服器
        print("Waitress 未安裝，回退到 Flask 開發伺服器...")
        app.run(host="0.0.0.0", port=port, debug=False)
