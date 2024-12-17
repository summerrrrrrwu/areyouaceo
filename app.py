import os
import joblib
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler

# 初始化 Flask 應用
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created uploads folder at: {UPLOAD_FOLDER}")

# 初始化 Mediapipe Face Mesh 模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# 加載模型和 StandardScaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # 確保你保存過標準化的 scaler

def extract_face_landmarks(image_path):
    """提取圖片的臉部 478 特徵點"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None

    # 圖片縮放與轉換
    new_width = 256
    scaling_factor = new_width / image.shape[1]
    new_height = int(image.shape[0] * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)


    # 使用 Face Mesh 提取面部特徵
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        print("Error: No face detected.")
        return None

    face_landmarks = results.multi_face_landmarks[0].landmark
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])

    # 計算 42 維度特徵
    features = calculate_42_features(landmarks)
    return features

# 計算 42 維度特徵邏輯
def calculate_42_features(landmarks):
    try:
        features = []
        # 1. 臉寬、高、長寬比
        face_width = np.linalg.norm(landmarks[127] - landmarks[356])  # 左右臉寬
        face_length = np.linalg.norm(landmarks[10] - landmarks[152])  # 上下臉長
        face_ratio = face_length / face_width if face_width != 0 else 0

        # 2. 鼻翼寬度與鼻頭寬度的比例
        nose_wing_width = np.linalg.norm(landmarks[245] - landmarks[465])  # 鼻翼左右
        nose_tip_width = np.linalg.norm(landmarks[48] - landmarks[278])    # 鼻頭寬度
        nose_wing_to_tip_ratio = nose_wing_width / nose_tip_width if nose_tip_width != 0 else 0

        # 3. 三庭比例
        forehead_to_eyebrow_distance = np.linalg.norm(landmarks[10] - landmarks[9])
        eyebrow_to_nose_distance = np.linalg.norm(landmarks[9] - landmarks[94])
        nose_to_chin_distance = np.linalg.norm(landmarks[94] - landmarks[152])

        # 4. 眼寬與臉寬比例
        eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
        five_eye_ratio = face_width / (5 * eye_width) if eye_width != 0 else 0

        # 5. 眼長寬比例
        eye_height = np.linalg.norm(landmarks[386] - landmarks[374])
        eye_length_to_width_ratio = eye_width / eye_height if eye_height != 0 else 0

        # 6. 鼻子與臉面積比例
        nose_area = nose_wing_width * nose_tip_width
        face_area = face_length * face_width
        nose_to_face_area_ratio = nose_area / face_area if face_area != 0 else 0

        # 7. 鼻子到上唇與下唇到下巴比例
        nose_to_upper_lip_distance = np.linalg.norm(landmarks[90] - landmarks[0])
        lower_lip_to_chin_distance = np.linalg.norm(landmarks[17] - landmarks[152])
        nose_to_lip_ratio = nose_to_upper_lip_distance / lower_lip_to_chin_distance if lower_lip_to_chin_distance != 0 else 0

        # 8. 顴骨到下巴比例
        cheekbone_distance = face_width
        chin_distance = np.linalg.norm(landmarks[377] - landmarks[148])
        cheekbone_to_chin_ratio = cheekbone_distance / chin_distance if chin_distance != 0 else 0

        # 9. 眉毛寬度與眼寬比例
        eyebrow_width = np.linalg.norm(landmarks[285] - landmarks[300])
        eyebrow_to_eye_ratio = eyebrow_width / eye_width if eye_width != 0 else 0

        # 10. 鼻子寬度與嘴巴寬度比例
        mouth_width = np.linalg.norm(landmarks[61] - landmarks[291])
        nose_to_mouth_ratio = nose_tip_width / mouth_width if mouth_width != 0 else 0

        # 11. 眼尾角度
        eye_tail_vector = landmarks[263] - landmarks[362]
        eye_tail_angle = np.degrees(np.arccos(eye_tail_vector[0] / np.linalg.norm(eye_tail_vector)))

        # 12. 眉毛角度
        eyebrow_vector = landmarks[300] - landmarks[285]
        eyebrow_angle = np.degrees(np.arccos(eyebrow_vector[0] / np.linalg.norm(eyebrow_vector)))

        # 13. 額頭寬度與臉長比例
        forehead_width = np.linalg.norm(landmarks[54] - landmarks[284])
        forehead_face_length_ratio = forehead_width / face_length if face_length != 0 else 0

        # 14. 黑眼珠面積與眼睛面積比例
        iris_length = np.linalg.norm(landmarks[474] - landmarks[475])
        iris_area = iris_length ** 2  # 假設圓形近似
        eye_area = eye_width ** 2
        iris_eye_area_ratio = iris_area / eye_area if eye_area != 0 else 0

        # 15. 眉毛三角形面積與臉面積比例
        side_a = np.linalg.norm(landmarks[300] - landmarks[285])
        side_b = np.linalg.norm(landmarks[334] - landmarks[285])
        side_c = np.linalg.norm(landmarks[334] - landmarks[300])
        semi_perimeter = (side_a + side_b + side_c) / 2
        eyebrow_triangle_area = np.sqrt(semi_perimeter * (semi_perimeter - side_a) *
                                        (semi_perimeter - side_b) * (semi_perimeter - side_c))
        eyebrow_to_face_area_ratio = eyebrow_triangle_area / face_area if face_area != 0 else 0
        
        # 16. 眼尾到太陽穴的距離與顴骨寬度比例
        temple_distance = np.linalg.norm(landmarks[389] - landmarks[359])
        eye_tail_cheekbone_ratio = temple_distance / cheekbone_distance if cheekbone_distance != 0 else 0

        # 17. 眉心寬度與額頭寬度比例
        brow_center_width = np.linalg.norm(landmarks[285] - landmarks[55])
        brow_forehead_ratio = brow_center_width / forehead_width if forehead_width != 0 else 0

        # 18. 上嘴唇厚度與嘴唇厚度比例
        upper_lip_thickness = np.linalg.norm(landmarks[13] - landmarks[0])
        lip_thickness = np.linalg.norm(landmarks[17] - landmarks[0])
        upper_lip_ratio = upper_lip_thickness / lip_thickness if lip_thickness != 0 else 0

        # 19. 鼻長與鼻頭寬度比例
        nose_length = np.linalg.norm(landmarks[8] - landmarks[2])
        nose_length_to_width_ratio = nose_length / nose_tip_width if nose_tip_width != 0 else 0
        
        # 組合特徵
        features.extend([
            face_width, face_length, face_ratio, nose_wing_to_tip_ratio, forehead_to_eyebrow_distance,
            eyebrow_to_nose_distance, nose_to_chin_distance, eye_width, five_eye_ratio, eye_height,
            eye_length_to_width_ratio, nose_to_face_area_ratio, nose_to_upper_lip_distance,
            lower_lip_to_chin_distance, nose_to_lip_ratio, cheekbone_distance, chin_distance,
            cheekbone_to_chin_ratio, eyebrow_width, eyebrow_to_eye_ratio, mouth_width,
            nose_to_mouth_ratio, eye_tail_angle, eyebrow_angle, forehead_face_length_ratio,
            iris_eye_area_ratio, eyebrow_to_face_area_ratio, eye_tail_cheekbone_ratio,
            brow_forehead_ratio, upper_lip_ratio, nose_length_to_width_ratio
        ])

        # 確保特徵數量達到 42
        while len(features) < 42:
            features.append(0)  # 不足時補零

        return np.array(features[:42])

    except Exception as e:
        print(f"Error calculating features: {e}")
        return np.zeros(42)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "請選擇一個檔案！"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "沒有檔案被選擇！"})

    # 保存圖片
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    # 提取特徵
    features = extract_42_features(file_path)
    if features is None:
        return jsonify({"error": "無法檢測到臉部，請使用正確的臉部圖片！"})

    # 檢查特徵長度是否為 42
    if len(features) != 42:
        return jsonify({"error": "特徵提取錯誤，請使用其他圖片！"})

    # 標準化特徵
    features_scaled = scaler.transform([features])

    # 預測
    prediction_proba = model.predict_proba(features_scaled)[0]
    ceo_probability = prediction_proba[1] * 100

    # 回傳結果
    return jsonify({
        "CEO Probability": f"{ceo_probability:.2f}%",
        "Top Recommendations": ["建議：微笑一下，提升自信！", "建議：注意臉部線條，增強氣質！"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
