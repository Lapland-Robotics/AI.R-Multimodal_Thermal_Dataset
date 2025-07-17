import os
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

def analyze_rgb_image(image_path):
    results = model(image_path)  # or use a numpy array
    print_summary(results, image_path)

def analyze_thermal_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # pseudo-color for YOLO
    results = model(img_rgb)
    print_summary(results, image_path)

def print_summary(yolo_result, image_path):
    class_names = yolo_result[0].names  # dict mapping class indices to names
    class_ids = yolo_result[0].boxes.cls.cpu().numpy().astype(int)  # get class IDs of detections
    counts = Counter([class_names[i] for i in class_ids])
    summary = ', '.join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items())
    print(f"{image_path}: {summary}")

    result_img = yolo_result[0].plot()
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_filename)
    custom_name = base_name + "_yolo.jpg"
    save_path = os.path.join(image_dir, custom_name)
    cv2.imwrite(save_path, result_img)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'exemplary_subset')
    folder_list = os.listdir(data_dir)
    for folder in folder_list:
        image_path = data_dir+"/"+folder+"/scaled_left_rgb.png"
        analyze_rgb_image(image_path)
        image_path = data_dir+"/"+folder+"/thermal.png"
        analyze_thermal_image(image_path)


if __name__ == "__main__":
    main()

