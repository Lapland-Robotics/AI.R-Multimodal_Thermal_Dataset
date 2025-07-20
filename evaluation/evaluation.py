import os
import cv2
from MyPrediction import MyPredictionClass
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import logging
import torch
from torchvision import ops
from collections import Counter

logging.getLogger("ultralytics").setLevel(logging.ERROR)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

def analyze_rgb_image(image_path):
    results = model(image_path)
    save_image(results, image_path)
    return results

def analyze_thermal_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    results = model(img_rgb)
    save_image(results, image_path)
    return results

def save_image(yolo_result, image_path):
    class_names = yolo_result[0].names  # dict mapping class indices to names
    class_ids = yolo_result[0].boxes.cls.cpu().numpy().astype(int)  # get class IDs of detections
    counts = Counter([class_names[i] for i in class_ids])
    summary = ', '.join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items())
    # print(f"Summary : {summary}")

    result_img = yolo_result[0].plot()
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_filename)
    custom_name = base_name + "_obj_detected.jpg"
    save_path = os.path.join(image_dir, custom_name)
    cv2.imwrite(save_path, result_img)

def readfile(file_path):
    result = {}
    baseline_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    if cls == 1:
                        cls = 0
                    xywhn = [float(v) for v in parts[1:5]]
                    tensor = torch.tensor([xywhn], dtype=torch.float32)
                    
                    if cls not in result:
                        result[cls] = []
                    result[cls].append(xywhn)
                    baseline_count += 1

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
    
    return result, baseline_count

def xywhn_to_xyxy_tensor(xywhn, img_width, img_height):

    cx, cy, w, h = xywhn
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device='cuda:0')

def compare(base_line, prediction):
    height, width = prediction[0].orig_shape
    prediction_list = []
    abc = prediction[0].boxes
    for cls, box, score in zip(abc.cls, abc.xyxy, abc.conf):
        obj = MyPredictionClass(
            cls=int(cls.item()),
            box=box,
            score=score.item()
        )
        prediction_list.append(obj)
    
    if base_line and prediction_list:
        for key in base_line:
            for val in base_line[key]:
                baseline_box = xywhn_to_xyxy_tensor(val, width, height)
                best_iou = 0
                best_prediction = None
                for i in range(len(prediction_list)):
                    if prediction_list[i].cls == key:
                        iou_tensor = ops.box_iou(baseline_box, prediction_list[i].box.unsqueeze(0))
                        iou = iou_tensor.item()  # Convert to scalar
                        if iou > 0.5 and iou > best_iou:
                            best_iou = iou
                            best_prediction = prediction_list[i]
                            prediction_list[i].iou = iou 

                print(f"\t\t class {key}, box {baseline_box.squeeze().cpu().numpy().tolist()}, IOU: {best_iou:.4f}".expandtabs(4))
                print(f"\t\t matched prediction: {best_prediction}".expandtabs(4))


def getFilteredCount(predictions, condition=lambda x: x is not None):
    boxes = predictions[0].boxes
    target_classes = [0, 2, 5]
    count = sum(1 for cls in boxes.cls if int(cls) in target_classes)
    return count

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'exemplary_subset_labeled')
    folder_list = os.listdir(data_dir)

    for folder in folder_list:
        print(f"Processing folder: {folder}")

        path = data_dir+"/"+folder+"/scaled_left_rgb"
        base_line_rgb, baseline_count_rgb = readfile(path+".txt")
        predict_rgb = analyze_rgb_image(path+".jpg")
        predict_count_rgb = getFilteredCount(predict_rgb)
        print(f"\t RGB Predicted count: {predict_count_rgb}, Baseline count: {baseline_count_rgb}".expandtabs(4))
        compare(base_line_rgb, predict_rgb)

        path = data_dir+"/"+folder+"/thermal"
        base_line_thermal, baseline_count_thermal = readfile(path+".txt")
        predict_thermal = analyze_rgb_image(path+".jpg")
        predict_count_thermal = getFilteredCount(predict_thermal)
        print(f"\t thermal Predicted count: {predict_count_thermal}, Baseline count: {baseline_count_thermal}".expandtabs(4))
        compare(base_line_thermal, predict_thermal)
        print("---------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()

