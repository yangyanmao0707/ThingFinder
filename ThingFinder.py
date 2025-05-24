from ultralytics import YOLO
import cv2
import torch
import clip
from PIL import Image
import os

# 設備判斷
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入模型
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

def main():
    file_path = r"C:\SIDE_PROJECT\ThingFinder\sample.jpg"

    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案：{file_path}")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("❌ 圖片讀取失敗")
        return

    results = yolo_model(img)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        if label == "bird":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            penguin_label = classify_penguin(crop)
            print(f"YOLO辨識: bird, CLIP判斷: {penguin_label}")
            label = penguin_label

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("ThingFinder - CLIP enhanced", img)
    print("📌 按任意鍵關閉視窗")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
