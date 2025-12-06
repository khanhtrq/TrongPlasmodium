from ultralytics import YOLO
import argparse
import os
import cv2
import json
import torch
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from tkinter import filedialog
from argparse import Namespace
import threading

from Khanh_inference_simple import run_simple_inference

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt

import textwrap
from tqdm import tqdm


def inference(args):
    class_names = ["Ring", "Trophozoite", "Schizont", "Gametocyte", "Healthy"]
    parasite_names = ["Ring", "Trophozoite", "Schizont", "Gametocyte"]
    parasitemia_count = {"Ring": 0, "Trophozoite": 0, "Schizont": 0, 
                "Gametocyte": 0, "Healthy": 0}
    # --------------
    # RBCs Detection
    # --------------
    detection_model = YOLO(args.detection_model)

    detection_results = detection_model.predict(source=args.image_folder, save= True, 
                                                save_txt= True, save_conf= True)

    save_dir = detection_results[0].save_dir
    txt_result_dir = os.path.join(save_dir, "labels")
    txt_file_list = [f for f in os.listdir(txt_result_dir) if os.path.isfile(os.path.join(txt_result_dir, f))]
    print("Number of blood smear images:", len(txt_file_list))

    for txt_file in txt_file_list:
        img_name = [f for f in os.listdir(args.image_folder) if f.startswith(txt_file.split('.')[0])][0]
        img_path = os.path.join(args.image_folder, img_name)
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        output_folder = os.path.join(save_dir, 'crop', txt_file.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "dummy_label"), exist_ok=True)
        print("Save cropped RBCs to:", output_folder)

        cell_detection_result_file = os.path.join(txt_result_dir, txt_file)
        with open(cell_detection_result_file, "r") as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()

            class_name, x_center, y_center, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)
            
            # Get top-left and bottom-right coordinates
            x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
            x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
            
            # Crop object
            cropped_object = image[y1:y2, x1:x2]

            # output_filename = os.path.join(output_folder, f"{i+1}.jpg")
            output_filename = os.path.join(output_folder, "dummy_label", f"{i+1}.jpg")
            cv2.imwrite(output_filename, cropped_object)

    # ---------------
    # CLASSIFICATION
    # ---------------
    detection_save_dir = save_dir

    model_checkpoint = args.cls_model
    model_name = 'efficientnet_b1.ra4_e3600_r240_in1k'
    model_num_classes = 5
    # --------------
    txt_result_dir = os.path.join(detection_save_dir, "labels")

    print("Classification progress:")
    for rbc_folder in tqdm(os.listdir(os.path.join(detection_save_dir, 'crop'))):
        # Read blood smear image
        blood_image_path = glob.glob(f"{args.image_folder}/{rbc_folder}*")[0]
        blood_image = cv2.imread(blood_image_path)

        folder_path = os.path.join(os.path.join(detection_save_dir, 'crop'), rbc_folder)
        input_images = []
        for root, _, files in os.walk(folder_path):        
            files.sort(key=lambda x: int(x.split('.')[0]))
            for file in files:
                input_images.append(os.path.abspath(os.path.join(root, file)))

        os.makedirs(f'./runs/classification/{rbc_folder}', exist_ok= True)

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
            imgf_root = folder_path
        )
        
        txt_file = [f for f in os.listdir(os.path.join(detection_save_dir, "labels")) if f.startswith(rbc_folder)][0]
        cell_detection_result_file = os.path.join(txt_result_dir, txt_file)
        
        os.makedirs(os.path.join(args.save_dir, "results/labels"), exist_ok=True)
        cls_detection_result = os.path.join(os.path.join(args.save_dir, "results/labels"), txt_file)
        cls_detection_result_list = []

        # detection result, format: Saves detection resut
        # format: [class] [x_center] [y_center] [width] [height] [confidence] 
        with open(cell_detection_result_file, "r") as file:
            lines = file.readlines()

        #list of predictions to compute confusion matrix
        pred_conf = []
        image_path = []
        
        colors = [
            "#006400",  # dark green
            "#00ff00",  # neon green
            "#ff7f00",  # strong orange
            "#0080ff",  # vivid blue
            "#6e6e6e"   # darker gray replacing #bfc0c2
        ]
        # ["#74a685", "#a4fea5", "#f7bd83", "#7adcef", "#bfc0c2"]
        # format: [class] [x_center] [y_center] [width] [height] [confidence] 
        for i, line in enumerate(lines):
            parts = line.strip().split()
            class_name, x_center, y_center, w, h, conf_score = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)

            # <class_name> <confidence> <left> <top> <right> <bottom>      
            x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
            x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
            # cls_class_id = classification_results[i]['pred_label']
            # pred_score = classification_results[i]['pred_score']

            cls_class_id = classification_results['inference_results']['predictions'][i] # classification_results[i].pred_label.item()
            pred_score = classification_results['inference_results']['confidences'][i] # classification_results[i].pred_score
            
            cls_detection_result_list.append('{} {} {} {} {} {}\n'.format(cls_class_id, conf_score, x1, y1, x2, y2))

            # Drawing bounding box
            # Should be prediction score (confidence score of the classification)

            hex_color = colors[cls_class_id]
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            label_text = f"{class_names[cls_class_id]} {conf_score:.2f}"
            
            if cls_class_id == 4:
                cv2.rectangle(blood_image, (x1, y1), (x2, y2), bgr, 3)
                thickness = 3
                cv2.putText(blood_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), thickness)
            else:
                cv2.rectangle(blood_image, (x1, y1), (x2, y2), bgr, 6)
                thickness = 4
                cv2.putText(blood_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness)

            stage_name = class_names[cls_class_id]
            parasitemia_count[stage_name] += 1

        # Save the refined result
        with open(cls_detection_result, "w") as refined_file:
            for line in cls_detection_result_list:
                refined_file.write(line)
        # Save annotated image
        os.makedirs(os.path.join(args.save_dir, "results", os.path.split(blood_image_path)[0]), exist_ok=True)
        cv2.imwrite(os.path.join(args.save_dir, "results", blood_image_path), blood_image)

    print(parasitemia_count)

    n_cells = sum(parasitemia_count[stage] for stage in parasitemia_count.keys())
    percentage_all = {}
    for stage in parasitemia_count.keys():
        percentage_all[stage] = parasitemia_count[stage] / n_cells

    percentage_parasite = {}
    n_parasite = sum(parasitemia_count[stage] for stage in parasite_names)
    for stage in parasite_names:
        percentage_parasite[stage] = parasitemia_count[stage] / n_parasite

    parasitemia = {"count": parasitemia_count,
                "percentage": percentage_all,
                "percentage_parasite": percentage_parasite}
    with open('./runs/parasitemia.json', 'w') as f:
        json.dump(parasitemia, f, indent=4)

    return parasitemia

# ---------------
# Arguments UI
# ---------------
def browse_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def browse_file(entry):
    file = filedialog.askopenfilename()
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

def get_args_gui(root):
    """Tkinter GUI for classification pipeline args. Returns argparse.Namespace"""

    args = {}

    def run_pipeline():
        nonlocal args
        args = Namespace(
            image_folder=image_folder_entry.get(),
            detection_model=detection_model_entry.get(),
            # cls_config=cls_config_entry.get(),
            cls_model=cls_model_entry.get(),
            save_dir=save_dir_entry.get(),
            conf_threshold= 0.7, # float(conf_threshold_entry.get()),
            iou_threshold= 0.5, #float(iou_threshold_entry.get()),
            # cls_batch_size=int(cls_batch_size_entry.get()),
            num_classes=5 # int(num_classes_entry.get())
        )
        input_win.destroy()

    input_win = tk.Toplevel(root)
    input_win.title("Classification Pipeline Config")

    # helper row builder
    def make_row(label, row, browse_type=None, default_val=None):
        tk.Label(input_win, text=label).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(input_win, width=50)
        entry.grid(row=row, column=1)
        if default_val is not None:
            entry.insert(0, str(default_val))
        if browse_type == "folder":
            tk.Button(input_win, text="Browse", command=lambda: browse_folder(entry)).grid(row=row, column=2)
        elif browse_type == "file":
            tk.Button(input_win, text="Browse", command=lambda: browse_file(entry)).grid(row=row, column=2)
        return entry
    
    def make_dropdown_row(label, row, options, default_val=None):
        tk.Label(input_win, text=label).grid(row=row, column=0, sticky="w")
        combo = ttk.Combobox(input_win, values=options, width=47)
        combo.grid(row=row, column=1)
        if default_val:
            combo.set(default_val)
        return combo

    detection_models = ["model/yolo12m_reco.pt", "model/yolo12s_reco.pt", 
                        "model/yolo11m_reco.pt", "model/yolo11s_reco.pt",
                        "model/yolo10s_reco.pt"]
    classification_models = ["model/final_plasmodium.pth"]

    # --- Inputs with defaults ---
    image_folder_entry    = make_row("Bloodsmear images folder:", 0, "folder", "images")
    save_dir_entry        = make_row("Save output to:", 1, "folder", "runs")
    detection_model_entry = make_dropdown_row("Detection model:", 2, detection_models, "model/yolo12m_reco.pt")
    cls_model_entry = make_dropdown_row("Classification model:", 3, classification_models, "model/final_plasmodium.pth")

    # Run button
    tk.Button(input_win, text="Run", command=run_pipeline, bg="lightgreen").grid(row=9, columnspan=3, pady=10)


    user_choice = {"retry": True}
    def exit_processing():
        user_choice["retry"] = False
        input_win.destroy()
    input_win.protocol("WM_DELETE_WINDOW", exit_processing)
    input_win.grab_set()
    root.wait_window(input_win)

    return args, user_choice["retry"]

# -------------------
#      Result UI 
# -------------------
def show_results_ui(root, args, folder, parasitemia = None):
    # Collect images
    exts = (".jpg", ".png", ".jpeg", ".bmp")
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    image_files.sort()

    if not image_files:
        print("No images found in", folder)
        return
    res_win = tk.Toplevel(root)
    res_win.title("Results")
    res_win.geometry("1080x750+70+10")


    # --- Top Frame (Text + Pie Chart) ---
    top_frame = tk.Frame(res_win)
    top_frame.pack(fill="x", padx=10, pady=5)

    # Info text at top
    info_label = tk.Label(res_win, text="Plasmodium Development Stage Detection", font=("Arial", 12), anchor="w", justify="left")
    info_label.pack(pady=5)

    parasite_percent = sum([parasitemia['percentage'][para_name] for para_name in ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte']])
    para_text = f"Parasite percentage: {parasite_percent*100:14.2f}%"
    parasite_count = sum([parasitemia['count'][para_name] for para_name in ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte']])

    parasitemia_text = (
        f"Ring:        {parasitemia['percentage']['Ring']*100:14.2f}% {parasitemia['percentage_parasite']['Ring']*100:15.2f}% ({parasitemia['count']['Ring']}/{parasite_count}) {parasitemia['count']['Ring']:10} \n"
        f"Trophozoite: {parasitemia['percentage']['Trophozoite']*100:14.2f}% {parasitemia['percentage_parasite']['Trophozoite']*100:15.2f}% ({parasitemia['count']['Trophozoite']}/{parasite_count}) {parasitemia['count']['Trophozoite']:10} \n"
        f"Schizont:    {parasitemia['percentage']['Schizont']*100:14.2f}% {parasitemia['percentage_parasite']['Schizont']*100:15.2f}% ({parasitemia['count']['Schizont']}/{parasite_count}) {parasitemia['count']['Schizont']:10} \n"
        f"Gametocyte:  {parasitemia['percentage']['Gametocyte']*100:14.2f}% {parasitemia['percentage_parasite']['Gametocyte']*100:15.2f}% ({parasitemia['count']['Gametocyte']}/{parasite_count}) {parasitemia['count']['Gametocyte']:10} \n"
        f"Healthy:     {parasitemia['percentage']['Healthy']*100:14.2f}%  {parasitemia['count']['Healthy']:33}"
    )
    parasitemia_text =  " "*14 + "Percentage(all)" + " "* 2 + "Percentage(parasites)" +  " "* 6 + "Count\n" + parasitemia_text
    print(parasitemia_text)

    text = f"""Result images saved to: {folder}\n\n{para_text}\n\n{parasitemia_text}"""
    lines = text.split("\n")  # split by existing newlines
    wrapped_lines = []
    for line in lines:
        # wrap each line to ~70 characters
        wrapped_lines.extend(textwrap.wrap(line, width=70) or [""])  
    # join back with \n
    text = "\n".join(wrapped_lines)

    text_label = tk.Label(
        top_frame,
        text=text,
        font=("Courier", 10),
        anchor="w",
        justify="left",
        bg="lightgray",
        width=72,
        height=12,
    )
    text_label.pack(side="left", padx=5, pady=5)

    # -------------
    # parasite pie chart
    # -------------
    fig1 = pie_chart(parasitemia)
    fig1.savefig(os.path.join(args.save_dir, "pie_parasite.png"), bbox_inches='tight')

    # -------------
    # all pie chart
    # -------------
    fig2 = pie_chart(
        parasitemia,
        labels=["Ring", "Trophozoite", "Schizont", "Gametocyte", "Healthy"],
        title='Percentage (all)'
    )
    fig2.savefig(os.path.join(args.save_dir, "pie_all.png"), bbox_inches='tight')

    img_pie_para = Image.open(os.path.join(args.save_dir, "pie_parasite.png"))
    img_pie_all = Image.open(os.path.join(args.save_dir, "pie_all.png"))
    total_width = img_pie_para.width + img_pie_all.width
    max_height = max(img_pie_para.height, img_pie_all.height)

    pie_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    pie_img.paste(img_pie_para, (0, 0))
    pie_img.paste(img_pie_all, (img_pie_para.width, 0))

    fig, ax = plt.subplots()
    ax.imshow(pie_img)
    ax.axis("off")
    fig.savefig(os.path.join(args.save_dir, "combined.png"), bbox_inches='tight')

    image = Image.open(os.path.join(args.save_dir, "combined.png"))
    width, height = image.size
    ratio = 0.85
    image = image.resize((int(width*ratio), int(height*ratio)))
    photo = ImageTk.PhotoImage(image)

    # Create a Label widget to display the image
    chart_label = tk.Label(top_frame, image=photo)
    chart_label.image = photo  # keep a reference!
    chart_label.pack(side="right", padx=20, pady=20)

    index = {"cur": 0}  # mutable container to allow update inside function
    def resize_keep_aspect(img, max_size=(600, 400)):
        """Resize image to fit in max_size, keeping aspect ratio"""
        img.thumbnail(max_size, Image.Resampling.LANCZOS)  # in-place resize
        return img

    def show_image():
        img = Image.open(image_files[index["cur"]])
        img = resize_keep_aspect(img, (1000, 430))  # resize to fit
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img
        res_win.title(f"Result Viewer - {os.path.basename(image_files[index['cur']])}")

    def next_img():
        index["cur"] = (index["cur"] + 1) % len(image_files)
        show_image()

    def prev_img():
        index["cur"] = (index["cur"] - 1) % len(image_files)
        show_image()

    # Frame that holds BOTH image and buttons
    display_frame = tk.Frame(res_win)
    display_frame.pack(pady=10)

    # Image on the left
    img_label = tk.Label(display_frame)
    img_label.pack(side="left", padx=10)

    # Right frame for buttons + extra controls
    right_frame = tk.Frame(display_frame)
    right_frame.pack(side="right", padx=10, fill="y")  # fill vertically

    # Top buttons
    btn_frame = tk.Frame(right_frame)
    btn_frame.pack(side="top")  # top of right_frame
    prev_btn = ttk.Button(btn_frame, text="<< Previous", command=prev_img)
    prev_btn.pack(pady=5)
    next_btn = ttk.Button(btn_frame, text="Next >>", command=next_img)
    next_btn.pack(pady=5)


    # Show first image
    show_image()
    
    user_choice = {"retry": False}

    def continue_processing():
        user_choice["retry"] = True
        res_win.destroy()

    def exit_processing():
        user_choice["retry"] = False
        res_win.destroy()

    # Bottom frame (e.g., for control buttons)
    bottom_frame = tk.Frame(right_frame)
    bottom_frame.pack(side="bottom", pady=20)  # bottom of right_frame
    tk.Button(bottom_frame, text="Process Another Folder", command=continue_processing, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(bottom_frame, text="Exit", command=exit_processing, bg="lightcoral").pack(side="left", padx=5)
    
    res_win.protocol("WM_DELETE_WINDOW", exit_processing)
    res_win.grab_set()
    root.wait_window(res_win)  # Wait until user closes the results window

    return user_choice["retry"]

def pie_chart(parasitemia, labels= ["Ring", "Trophozoite", "Schizont", "Gametocyte"],
              title= 'Percentage (parasites)'):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    if "Healthy" not in labels:
        sizes = [parasitemia["percentage_parasite"][n] for n in labels]
        colors = [
            "#006400",  # dark green
            "#00ff00",  # neon green
            "#ff7f00",  # strong orange
            "#0080ff",  # vivid blue
        ]
        high_contrast_colors = [
            "#006400",  # dark green
            "#00ff00",  # neon green
            "#ff7f00",  # strong orange
            "#0080ff",  # vivid blue
            "#000000"   # black
        ]
    else:
        sizes = [parasitemia["percentage"][n] for n in labels]
        colors = [
            "#006400",  # dark green
            "#00ff00",  # neon green
            "#ff7f00",  # strong orange
            "#0080ff",  # vivid blue
            "#6e6e6e"   # darker gray replacing #bfc0c2
        ]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,         # start from top
        counterclock=False,    # move clockwise
        radius=1.3,
        colors=colors,
        wedgeprops=dict(width=0.5)
    )

    # Hide labels and texts for zero portions
    for i, size in enumerate(sizes):
        if size <= 0.05:
            texts[i].set_visible(False)
            autotexts[i].set_visible(False)

    # Style
    for text in texts:
        text.set_horizontalalignment('center')
        text.set_fontsize(16)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(16)

    if "Healthy" in labels:
        legend = ax.legend(labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(legend.get_texts(), fontsize=14)
        plt.setp(legend.get_title(), fontsize=14)

    ax.set_title(title, fontsize=20)

    return fig

# ----------------
# running UI
# ----------------
def show_processing_window(root, args):
    """Show a 'Processing...' window while running inference(args) without threading."""
    result = {"parasite_return": None}

    # Create a small Toplevel window
    proc_win = tk.Toplevel(root)
    proc_win.title("Processing")
    proc_win.geometry("250x80")
    proc_win.resizable(False, False)

    tk.Label(proc_win, text="Processing...", font=("Arial", 12)).pack(expand=True, pady=20)

    def worker():
        # heavy computation runs here
        result["parasite_return"] = inference(args)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Poll for thread completion
    def check_thread():
        if thread.is_alive():
            proc_win.after(100, check_thread)  # check again after 100ms
        else:
            proc_win.destroy()  # safely destroy from main thread

    user_choice = {"retry": True}
    def exit_processing():
        user_choice["retry"] = False
        proc_win.destroy()
        result["parasite_return"] = None

    proc_win.protocol("WM_DELETE_WINDOW", exit_processing)

    proc_win.after(100, check_thread)
    proc_win.grab_set()         # make modal
    root.wait_window(proc_win)  # wait until proc_win is destroyed

    return result["parasite_return"], user_choice["retry"]



def main_loop():
    root = tk.Tk()
    root.withdraw()  # hide main root
    
    continue_processing = True
    while continue_processing:
        # Get GUI args
        args, continue_processing = get_args_gui(root)  # This creates a new Tk instance safely
        if not continue_processing:
            break

        # Run processing window (threaded)
        parasitemia, continue_processing = show_processing_window(root, args)
        if not continue_processing:
            break
    
        # Show results window
        continue_processing = show_results_ui(
            root,
            args=args,
            folder=os.path.join(args.save_dir, "results", args.image_folder),
            parasitemia=parasitemia
        )
        # retry = messagebox.askyesno("Continue?", "Do you want to process another folder?")
        # if not retry:
        #     break

    root.destroy()

if __name__ == "__main__":
    main_loop()
