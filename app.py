from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from PIL import Image
import cv2

app = Flask(__name__)

# 載入模型
model = joblib.load("stacking_model.pkl")  # 你訓練好的模型
important_features = ['鼻子寬度', '下巴長度', '眼角距離']  # 替換成你的前三重要特徵

@app.route("/")
def home():
    return "CEO Predictor is running!"
    
@app.route("/predict", methods=["POST"])
def predict():
    # 獲取圖片
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = np.array(image)

    # 圖片處理邏輯（假設你有固定輸入維度）
    processed_image = cv2.resize(image, (224, 224)).flatten().reshape(1, -1)  # 替換成模型輸入邏輯

    # 預測
    probability = model.predict_proba(processed_image)[0, 1] * 100  # CEO 機率

    # 搞笑醫美建議
    suggestions = [
        f"鼻子太寬啦！可以考慮醫美縮窄一下鼻翼。",
        f"下巴太短了，補個下巴立刻 CEO 氣場全開！",
        f"眼角距離不夠理想？開個眼角當 CEO 無人能擋！"
    ]
    response = {
        "probability": round(probability, 2),
        "suggestion": suggestions[np.argmax(processed_image)]  # 搞笑建議
    }
    return jsonify(response)

# 啟動 Flask 應用程式
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 從環境變數獲取 PORT
    app.run(host="0.0.0.0", port=port, debug=True)
