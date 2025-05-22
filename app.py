from flask import Flask, request, jsonify, send_file
import base64, cv2, os, numpy as np

from gan_anonymizer import GanAnonymizer
from foto_face_detector import PhotoFaceDetector
from video_face_detector import VideoFaceDetector

app = Flask(__name__)

# 1) GAN modelini yükle
gan = GanAnonymizer("saved_models/generator2.pth")

# 2) Fotoğraf ve video dedektörlerini oluştur
photo_detector = PhotoFaceDetector(anonymizer=gan)
video_detector = VideoFaceDetector(anonymizer=gan)

# === 1. Kameradan gelen base64 kare ===
@app.route('/anonymize/frame', methods=['POST'])
def anonymize_frame():
    data = request.json or {}
    base64_img = data.get("image", "").split(",")[-1]
    img_data    = base64.b64decode(base64_img)
    np_arr      = np.frombuffer(img_data, np.uint8)
    frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Sadece yüz ROI'lerini anonimleştir
    det, ano = video_detector.process_frame(frame)
    _, buf   = cv2.imencode('.jpg', ano)
    b64      = base64.b64encode(buf).decode('utf-8')
    return jsonify({"image": f"data:image/jpeg;base64,{b64}"})

# === 2. Yüklü fotoğrafı işle ===
@app.route('/anonymize/image', methods=['GET'])
def anonymize_image():
    in_path  = "uploads/input.jpg"
    out_path = "outputs/anonymized.jpg"
    if not os.path.exists(in_path):
        return jsonify({"error": "Image not found"}), 404

    img = cv2.imread(in_path)
    # PhotoFaceDetector bize (ano_img, det_img) döndürüyor
    ano_img, _ = photo_detector.process_image(img)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(out_path, ano_img)
    return send_file(out_path, mimetype='image/jpeg')

# === 3. Yüklü videoyu işle ===
@app.route('/anonymize/video', methods=['GET'])
def anonymize_video():
    in_path = "uploads/input_video.mp4"
    if not os.path.exists(in_path):
        return jsonify({"error": "Video not found"}), 404

    try:
        # process_video metodun headless olarak video dosyasına yazar
        out_path = video_detector.process_video(in_path)
    except Exception as e:
        return jsonify({"error": f"Video processing failed: {e}"}), 500

    return send_file(out_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(port=5000)
