import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

# 初始化 Flask 應用
app = Flask(__name__)

# 加載保存的 Random Forest 模型
model = joblib.load("random_forest_model.pkl")
scaler = StandardScaler()  # 假設使用 StandardScaler 標準化

# 前端頁面 (HTML, 上傳功能)
@app.route("/")
def index():
    return render_template("index.html")

# 預測端點
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 接收輸入數據
        uploaded_data = request.json["features"]
        input_features = np.array(uploaded_data).reshape(1, -1)

        # 標準化輸入數據
        input_scaled = scaler.fit_transform(input_features)

        # 預測
        prediction_proba = model.predict_proba(input_scaled)[0]
        result = {"CEO Probability": f"{prediction_proba[1]*100:.2f}%"}
        
        # 給出醫美建議 (搞笑版)
        recommendations = []
        if "right eyebrow triangle angles_1" in uploaded_data:
            recommendations.append("建議：提眉手術，眉尾高一點更有領導力！")
        if "triangle angles (eye tail, lower lip, center)_2" in uploaded_data:
            recommendations.append("建議：微笑訓練，嘴角上揚更 CEO！")

        result["Recommendations"] = recommendations

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# 啟動 Flask 應用程式
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 從環境變數獲取 PORT
    app.run(host="0.0.0.0", port=port, debug=True)
