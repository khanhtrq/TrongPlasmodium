import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ColorThresholdApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Threshold Filter")
        self.root.geometry("1200x800")
        
        # Danh sách các ảnh đã load
        self.image_paths = []
        self.original_images = []
        self.processed_images = []
        self.display_images = []
        
        self.mode = tk.StringVar(value="threshold")  # "threshold" or "kmeans"
        self.kmeans_n = tk.IntVar(value=3)
        
        self.hist_canvas = None
        self.hist_fig = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame chính
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame cho controls
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Button load nhiều ảnh
        load_btn = tk.Button(control_frame, text="Load Images", command=self.load_images, 
                            font=("Arial", 12), bg="#4CAF50", fg="white")
        load_btn.pack(pady=10, fill=tk.X)

        # Hiển thị tên file
        # Sliders cho RGB thresholds
        tk.Label(control_frame, text="Red Channel", font=("Arial", 10, "bold")).pack(pady=(20,5))
        self.red_min = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                               command=self.update_image, label="Min Red")
        self.red_min.pack(fill=tk.X, pady=2)
        
        self.red_max = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                               command=self.update_image, label="Max Red")
        self.red_max.set(255)
        self.red_max.pack(fill=tk.X, pady=2)
        
        tk.Label(control_frame, text="Green Channel", font=("Arial", 10, "bold")).pack(pady=(20,5))
        self.green_min = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 command=self.update_image, label="Min Green")
        self.green_min.pack(fill=tk.X, pady=2)
        
        self.green_max = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 command=self.update_image, label="Max Green")
        self.green_max.set(255)
        self.green_max.pack(fill=tk.X, pady=2)
        
        tk.Label(control_frame, text="Blue Channel", font=("Arial", 10, "bold")).pack(pady=(20,5))
        self.blue_min = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                command=self.update_image, label="Min Blue")
        self.blue_min.pack(fill=tk.X, pady=2)
        
        self.blue_max = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                command=self.update_image, label="Max Blue")
        self.blue_max.set(255)
        self.blue_max.pack(fill=tk.X, pady=2)
        
        # Reset button
        reset_btn = tk.Button(control_frame, text="Reset", command=self.reset_sliders,
                             font=("Arial", 10), bg="#f44336", fg="white")
        reset_btn.pack(pady=20, fill=tk.X)
        
        # Save button
        save_btn = tk.Button(control_frame, text="Save Result", command=self.save_image,
                            font=("Arial", 10), bg="#2196F3", fg="white")
        save_btn.pack(pady=5, fill=tk.X)
        
        # Thêm lựa chọn chế độ
        tk.Label(control_frame, text="Mode", font=("Arial", 10, "bold")).pack(pady=(10,2))
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        tk.Radiobutton(mode_frame, text="Color Threshold", variable=self.mode, value="threshold", 
                       command=self.update_image).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="KMeans", variable=self.mode, value="kmeans", 
                       command=self.update_image).pack(side=tk.LEFT)
        
        # Entry nhập số vùng cho KMeans
        kmeans_frame = tk.Frame(control_frame)
        kmeans_frame.pack(fill=tk.X, pady=(2,10))
        tk.Label(kmeans_frame, text="N (KMeans):").pack(side=tk.LEFT)
        kmeans_entry = tk.Entry(kmeans_frame, textvariable=self.kmeans_n, width=4)
        kmeans_entry.pack(side=tk.LEFT)
        
        # Frame cho ảnh
        image_frame = tk.Frame(main_frame, bg="gray")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas để hiển thị ảnh
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Thêm frame cho histogram
        hist_frame = tk.Frame(image_frame)
        hist_frame.pack(fill=tk.X, pady=(5, 0))
        self.hist_frame = hist_frame
        
        # Scrollbars
        h_scroll = tk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = tk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
    def load_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_paths:
            self.image_paths = list(file_paths)
            self.original_images = []
            for path in self.image_paths:
                try:
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.original_images.append(img)
                except Exception:
                    self.original_images.append(None)
            self.reset_sliders()
            self.update_image()

    def apply_color_threshold(self, img):
        if img is None:
            return None
        r_min, r_max = self.red_min.get(), self.red_max.get()
        g_min, g_max = self.green_min.get(), self.green_max.get()
        b_min, b_max = self.blue_min.get(), self.blue_max.get()
        mask_r = (img[:,:,0] >= r_min) & (img[:,:,0] <= r_max)
        mask_g = (img[:,:,1] >= g_min) & (img[:,:,1] <= g_max)
        mask_b = (img[:,:,2] >= b_min) & (img[:,:,2] <= b_max)
        final_mask = mask_r & mask_g & mask_b
        result = img.copy()
        result[~final_mask] = [0, 0, 0]
        return result

    def apply_kmeans(self, img, n_clusters):
        if img is None:
            return None
        Z = img.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            K = max(2, int(n_clusters))
        except Exception:
            K = 3
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((img.shape))
        return result

    def plot_histogram(self, img):
        # Xóa biểu đồ cũ nếu có
        if self.hist_canvas:
            self.hist_canvas.get_tk_widget().destroy()
            self.hist_canvas = None
        if img is None:
            return
        fig, ax = plt.subplots(figsize=(4,2), dpi=100)
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0,256])
            ax.plot(hist, color=color, label=f'{color.upper()}')
        ax.set_xlim([0,256])
        ax.set_title("RGB Histogram")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        self.hist_fig = fig
        self.hist_canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill=tk.X)
        plt.close(fig)

    def update_image(self, *args):
        # Xử lý tất cả ảnh và hiển thị trên canvas
        self.processed_images = []
        self.display_images = []
        if not self.original_images:
            self.canvas.delete("all")
            return
        max_size = 300  # Kích thước tối đa mỗi ảnh nhỏ
        for img in self.original_images:
            if self.mode.get() == "threshold":
                proc = self.apply_color_threshold(img)
            else:
                n = self.kmeans_n.get()
                proc = self.apply_kmeans(img, n)
            self.processed_images.append(proc)
            if proc is not None:
                h, w = proc.shape[:2]
                scale = min(max_size / max(h, w), 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                display_img = cv2.resize(proc, (new_w, new_h))
                pil_image = Image.fromarray(display_img)
                self.display_images.append(ImageTk.PhotoImage(pil_image))
            else:
                self.display_images.append(None)
        # Hiển thị tất cả ảnh trên canvas theo lưới
        self.canvas.delete("all")
        cols = 3
        pad = 10
        for idx, imgtk in enumerate(self.display_images):
            if imgtk is None:
                continue
            row = idx // cols
            col = idx % cols
            x = col * (max_size + pad)
            y = row * (max_size + pad)
            self.canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
            # Hiển thị tên file dưới mỗi ảnh
            if idx < len(self.image_paths):
                filename = os.path.basename(self.image_paths[idx])
                self.canvas.create_text(x + max_size//2, y + max_size + 12, text=filename, anchor=tk.N, font=("Arial", 9), fill="gray")
        # Giữ tham chiếu để tránh bị thu hồi bộ nhớ
        self.canvas.images = self.display_images
        # Cập nhật vùng scroll
        n_rows = (len(self.display_images) + cols - 1) // cols
        self.canvas.configure(scrollregion=(0, 0, cols * (max_size + pad), n_rows * (max_size + pad + 25)))
        # Vẽ histogram cho ảnh đầu tiên đã xử lý
        if self.processed_images and self.processed_images[0] is not None:
            self.plot_histogram(self.processed_images[0])
        else:
            self.plot_histogram(None)

    def reset_sliders(self):
        self.red_min.set(0)
        self.red_max.set(255)
        self.green_min.set(0)
        self.green_max.set(255)
        self.blue_min.set(0)
        self.blue_max.set(255)
    
    def save_image(self):
        if self.processed_images is None:
            messagebox.showwarning("Warning", "No processed image to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert RGB to BGR for OpenCV save
                save_img = cv2.cvtColor(self.processed_images[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, save_img)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot save image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorThresholdApp(root)
    root.mainloop()