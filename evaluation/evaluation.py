import os
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import logging
import torch
from torchvision import ops

logging.getLogger("ultralytics").setLevel(logging.ERROR)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

def analyze_rgb_image(image_path):
    results = model(image_path)  # or use a numpy array
    # print_summary(results, image_path)
    return results

def analyze_thermal_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # pseudo-color for YOLO
    results = model(img_rgb)
    # print_summary(results, image_path)
    return results

def print_summary(yolo_result, image_path):
    class_names = yolo_result[0].names  # dict mapping class indices to names
    class_ids = yolo_result[0].boxes.cls.cpu().numpy().astype(int)  # get class IDs of detections
    counts = Counter([class_names[i] for i in class_ids])
    summary = ', '.join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items())
    print(f"Summary for {image_path}: {summary}")

    # result_img = yolo_result[0].plot()
    # image_dir = os.path.dirname(image_path)
    # image_filename = os.path.basename(image_path)
    # base_name, _ = os.path.splitext(image_filename)
    # custom_name = base_name + "_yolo.jpg"
    # save_path = os.path.join(image_dir, custom_name)
    # cv2.imwrite(save_path, result_img)

def readfile(file_path):
    result = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    xywhn = [float(v) for v in parts[1:5]]
                    tensor = torch.tensor([xywhn], dtype=torch.float32)
                    
                    if cls not in result:
                        result[cls] = []
                    result[cls].append(tensor)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
    return result

def compare(base_line, prediction):

    for key in base_line:
        print(f"Key: {key}")
        for val in base_line[key]:
            print(f"  Value: {val}")
    # print("prediction : ")
    # print(prediction[0].boxes.xyxy)

    # iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
    # print('IOU : ', iou.numpy()[0][0])

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'exemplary_subset_labeled', 'darkness')
    folder_list = os.listdir(data_dir)

    for folder in folder_list:
        print(f"Processing folder: {folder}")
        path = data_dir+"/"+folder+"/scaled_left_rgb"
        base_line_rgb = readfile(path+".txt")
        predict_rgb = analyze_rgb_image(path+".jpg")
        compare(base_line_rgb, predict_rgb)

        path = data_dir+"/"+folder+"/thermal"
        base_line_thermal = readfile(path+".txt")
        predict_thermal = analyze_thermal_image(path+".jpg")
        # compare(base_line_thermal, predict_thermal)
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()

