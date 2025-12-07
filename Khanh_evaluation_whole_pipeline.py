from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import torch
# from mmpretrain.apis import ImageClassificationInferencer
# from pathlib import Path
from khanh_utils.confusion_matrix import DetectionConfusionMatrix

from Khanh_inference_simple import run_simple_inference


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--detection_model", type=str, help="detection model",
                    default=".\detect_results\yolo11m_reco.pt")
parser.add_argument("--blood_smear_images", type=str, 
                    default="draft_blood_smear",
                    help="path to folder with blood smear images")

# parser.add_argument("--cls_model", type=str, help="path to config file")
# parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")

parser.add_argument("--gt_folder", type=str, 
                    default="draft_blood_smear",
                    help="folder with annotation (entire pipeline)")

parser.add_argument('--conf_threshold', type=float, default=0.3, help="Confidence threshold for the whole pipeline")
parser.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for the whole pipeline")

parser.add_argument("--save_dir", type=str, 
                    default='runs/whole_pipeline_results',
                    help= "Directory to save confusion matrix.")
parser.add_argument("--cls_batch_size", type=int, default=32, help="batch size")

parser.add_argument('--merge_healthy_other', type=bool, default=False, help="Merge Healthy and Other class for evaluation")
parser.add_argument('--num_classes', type=int, default= 5, help="Number of classes for confusion matrix")




args = parser.parse_args()

# --------------
# RBCs Detection
# --------------

detection_model = YOLO(args.detection_model)

detection_results = detection_model.predict(source=args.blood_smear_images, save= True, 
                                            save_txt= True, save_conf= True, conf = args.conf_threshold)

save_dir = detection_results[0].save_dir
txt_result_dir = os.path.join(save_dir, "labels")
txt_file_list = [f for f in os.listdir(txt_result_dir) if os.path.isfile(os.path.join(txt_result_dir, f))]
print("Number of blood smear images:", len(txt_file_list))

for txt_file in txt_file_list:
    img_name = [f for f in os.listdir(args.blood_smear_images) if f.startswith(txt_file.split('.')[0])][0]
    img_path = os.path.join(args.blood_smear_images, img_name)
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    output_folder = os.path.join(save_dir, 'crop', txt_file.split('.')[0])
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "dummy_label"), exist_ok=True)
    print("Save cropped RBCs to:", output_folder)

    cell_detection_result_file = os.path.join(txt_result_dir, txt_file)
    with open(cell_detection_result_file, "r") as file:
        lines = file.readlines()
    valid_lines = []

    for i, line in enumerate(lines):
        parts = line.strip().split()

        class_name, x_center, y_center, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)
        
        # Get top-left and bottom-right coordinates
        x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
        x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
        
        # Crop object
        cropped_object = image[y1:y2, x1:x2]

        if cropped_object.size == 0:
            continue
        
        output_filename = os.path.join(output_folder, "dummy_label", f"{i+1}.jpg")
        cv2.imwrite(output_filename, cropped_object)
        valid_lines.append(line)
    
    with open(cell_detection_result_file, "w") as file:
        file.writelines(valid_lines)

# ---------------
# CLASSIFICATION
# ---------------
detection_save_dir = save_dir


# Nov 2: To be modified
# Replace with Trong's model
# --------------
# inferencer = ImageClassificationInferencer(
#     model = args.cls_model,
#     pretrained = args.cls_pretrained,
#     device='cuda')
model_checkpoint = "./model/final_plasmodium.pth"
model_name = 'efficientnet_b1.ra4_e3600_r240_in1k'
model_num_classes = 5
# --------------

txt_result_dir = os.path.join(detection_save_dir, "labels")

detection_conf_obj = DetectionConfusionMatrix(num_classes=args.num_classes, CONF_THRESHOLD=args.conf_threshold,
                                              IOU_THRESHOLD=args.iou_threshold)

for rbc_folder in tqdm(os.listdir(os.path.join(detection_save_dir, 'crop'))):
    #get the coordication of image
    # cell_img_name = [f for f in os.listdir(args.blood_smear_images) if f.startswith(rbc_folder)][0]
    # img_path = os.path.join(args.blood_smear_images, cell_img_name)
    # image = cv2.imread(img_path)

    #classification with mmpretrain model
    folder_path = os.path.join(os.path.join(detection_save_dir, 'crop'), rbc_folder, "dummy_label")
    input_images = []
    for root, _, files in os.walk(folder_path):        
        files.sort(key=lambda x: int(x.split('.')[0]))
        for file in files:
            input_images.append(os.path.abspath(os.path.join(root, file)))

    # classification_results = inferencer(inputs = input_images,
    #                                     show_dir = './visualize/',
    #                                     batch_size=args.cls_batch_size,
    #                                     return_datasamples=True)
    
    cropped_image_folder = os.path.join(os.path.join(detection_save_dir, 'crop'), rbc_folder)
    classification_results = run_simple_inference(
        model_name=model_name,  # Direct timm model name
        model_checkpoint=model_checkpoint,  # Direct path
        model_num_classes=model_num_classes,  # Explicitly specify model class count
        split='test',
        batch_size=16,
        config_path = 'config_prototype.yaml',
        save_scores=True,  # 💾 Enable softmax score saving
        scores_filename="test_scores_6cls_vs_7cls.txt",  # Custom filename
        run_phase2=True,  # 🔬 Enable Phase 2 evaluation
        verbose=False,
        imgf_root = cropped_image_folder
    )

    
    txt_file = [f for f in os.listdir(os.path.join(detection_save_dir, "labels")) if f.startswith(rbc_folder)][0]
    cell_detection_result_file = os.path.join(txt_result_dir, txt_file)
    
    os.makedirs(os.path.join(args.save_dir, "life_cycle_labels"), exist_ok=True)
    cls_detection_result = os.path.join(os.path.join(args.save_dir, "life_cycle_labels"), txt_file)
    cls_detection_result_list = []

    # detection result, format: Saves detection resut
    # format: [class] [x_center] [y_center] [width] [height] [confidence] 
    with open(cell_detection_result_file, "r") as file:
        lines = file.readlines()

    #list of predictions to compute confusion matrix
    pred_conf = []
    
    # format: [class] [x_center] [y_center] [width] [height] [confidence] 
    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_name, x_center, y_center, w, h, det_conf_score = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)

        # <class_name> <confidence> <left> <top> <right> <bottom>      
        x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
        x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
        # cls_class_name = classification_results[i]['pred_label']
        # pred_score = classification_results[i]['pred_score']

        # Nov 2: To be modified
        # --------------
        cls_class_name = classification_results['inference_results']['predictions'][i]
        conf_score = classification_results['inference_results']['confidences'][i]
        # input_images = classification_results['inference_results']['sample_paths'][i]
        # --------------

        if args.merge_healthy_other:
            # if label == other then label = healthy
            if cls_class_name == 5:
                cls_class_name = 4
        
        cls_detection_result_list.append('{} {} {} {} {} {}\n'.format(cls_class_name, conf_score, x1, y1, x2, y2))

        #predictions results to compute confusion matrix
        pred_conf.append([x1, y1, x2, y2, conf_score, cls_class_name])
    pred_conf = np.array(pred_conf, dtype = object)

    #Grounth truth, format <label>, <x1>, <y1>, <x2>, <y2>
    #list of grounth truth to compute confusion matrix

    gt_path = os.path.join(args.gt_folder, txt_file)
    gt_conf = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    for gt_sample in lines:
        gt_label, x1, y1, x2, y2 = gt_sample.split(' ')
        gt_label = int(gt_label)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gt_conf.append([gt_label, x1, y1, x2, y2])
    gt_conf = np.array(gt_conf, dtype = object)

    detection_conf_obj.process_batch(pred_conf, gt_conf, input_images)

    #Save the refined result
    with open(cls_detection_result, "w") as refined_file:
        for line in cls_detection_result_list:
           refined_file.write(line)

detection_conf = detection_conf_obj.return_matrix()
image_path = detection_conf_obj.return_image_path()
print(image_path)
with open(os.path.join(args.save_dir, 'image_path.json'), 'w') as f:
    json.dump(image_path, f, indent=4)

pr_metrics = detection_conf_obj.compute_PR_from_matrix(detection_conf)

print("Whole pipeline confusion matrix:")
print(torch.tensor(detection_conf, dtype=torch.int64))
print("Precision Recall:", pr_metrics)

#Save confusion matrix
os.makedirs(args.save_dir, exist_ok= True)
data = pr_metrics
data["confusion_matrix"] = detection_conf.tolist()
with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
    json.dump(data, f, indent=4)
