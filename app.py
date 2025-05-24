import os
import uuid
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import torch
import clip
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")

def classify_penguin(image_crop):
    image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image_pil).unsqueeze(0).to(device)

    text_labels = ["a penguin", "a bird", "not a penguin"]
    texts = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(texts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        return text_labels[indices[0].item()]

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("無法開啟影片檔案")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v 編碼器 (H264 也可用 'avc1')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    labels_set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if label == "bird":
                crop = frame[y1:y2, x1:x2]
                penguin_label = classify_penguin(crop)
                labels_set.add(penguin_label)
                label = penguin_label
            else:
                labels_set.add(label)

            text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return list(labels_set)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    result_video = None
    labels_detected = []
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "❌ 沒有檔案被上傳"
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "❌ 請選擇一張圖片或影片"
            return render_template("index.html", error=error)

        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1]

        # 讀入檔案 bytes
        file_bytes = file.read()
        np_file = np.frombuffer(file_bytes, np.uint8)

        if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # 圖片處理
            img = cv2.imdecode(np_file, cv2.IMREAD_COLOR)
            if img is None:
                error = "❌ 圖片讀取失敗"
                return render_template("index.html", error=error)

            results = yolo_model(img)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]

                if label == "bird":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]
                    penguin_label = classify_penguin(crop)
                    labels_detected.append(penguin_label)
                    label = penguin_label
                else:
                    labels_detected.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                text = f"{label} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            result_filename = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(UPLOAD_FOLDER, result_filename)
            cv2.imwrite(result_path, img)
            result_image = result_filename

        elif file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            # 影片處理
            # 暫存原始影片
            temp_video_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex}{file_ext}")
            with open(temp_video_path, "wb") as f:
                f.write(file_bytes)

            result_video_filename = f"result_{uuid.uuid4().hex}.mp4"
            result_video_path = os.path.join(UPLOAD_FOLDER, result_video_filename)

            try:
                labels_detected = process_video(temp_video_path, result_video_path)
                result_video = result_video_filename
            except Exception as e:
                error = f"❌ 影片處理失敗: {e}"
            finally:
                # 刪除暫存影片
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        else:
            error = "❌ 不支援的檔案格式，請上傳圖片或影片"

    return render_template("index.html", 
                           result_image=result_image,
                           result_video=result_video,
                           labels=labels_detected,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
