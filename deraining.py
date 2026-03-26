import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import sys

sys.path.append('MPRNet')
from MPRNet import MPRNet
import Utils

class DerainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Rain Removal System")
        self.root.state('zoomed')

        model_path = "model/baseline.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"The model weight file was not found: {model_path}\nPlease ensure that the file exists.")
            sys.exit(1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.model = MPRNet()
        Utils.load_checkpoint(self.model, model_path)
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.model.eval()
        print(f"The model has been loaded successfully: {model_path}")

        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        self.original_image = None
        self.derained_image = None
        self.original_image_tk = None
        self.derained_image_tk = None
        self.input_tensor = None

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        bottom_frame_height = screen_height // 4
        top_frame_height = screen_height - bottom_frame_height
        top_frame_width = screen_width // 2

        main_container = tk.PanedWindow(root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=0)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top: Image display area
        self.img_frame = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=0, height=top_frame_height)
        main_container.add(self.img_frame)

        # Left: Original image area
        self.left_frame = tk.Frame(self.img_frame, relief=tk.GROOVE, borderwidth=2)
        self.img_frame.add(self.left_frame, width=top_frame_width)
        tk.Label(self.left_frame, text="Rain Image (Input)", font=('Arial', 16, 'bold')).pack(pady=5)
        canvas_container_original = tk.Frame(self.left_frame)
        canvas_container_original.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_original = tk.Canvas(canvas_container_original, bg='lightgray')
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # Right: Deraining result area
        self.right_frame = tk.Frame(self.img_frame, relief=tk.GROOVE, borderwidth=2)
        self.img_frame.add(self.right_frame, width=top_frame_width)
        tk.Label(self.right_frame, text="Deraining Image (Output)", font=('Arial', 16, 'bold')).pack(pady=5)
        canvas_container_derained = tk.Frame(self.right_frame)
        canvas_container_derained.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_derained = tk.Canvas(canvas_container_derained, bg='lightgray')
        self.canvas_derained.pack(fill=tk.BOTH, expand=True)

        # Bottom: Control and Information area
        bottom_frame = tk.Frame(main_container, relief=tk.SUNKEN, borderwidth=2)
        main_container.add(bottom_frame)
        main_container.sash_place(0, 0, top_frame_height)

        # Bottom left: Button area
        btn_frame_bottom = tk.Frame(bottom_frame)
        btn_frame_bottom.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        btn_frame_bottom.grid_rowconfigure(0, weight=1)
        btn_frame_bottom.grid_rowconfigure(3, weight=1)

        self.btn_load = tk.Button(btn_frame_bottom, text="Select Image", command=self.load_image,
                                  width=15, height=2, bg='lightblue', font=('Arial', 10, 'bold'))

        self.btn_load.grid(row=1, column=0, padx=5, pady=2, sticky='n')

        self.btn_process = tk.Button(btn_frame_bottom, text="Deraining", command=self.process_image,
                                     state=tk.DISABLED, width=15, height=2, font=('Arial', 10, 'bold'))
        self.btn_process.grid(row=1, column=1, padx=5, pady=2, sticky='n')

        self.btn_save = tk.Button(btn_frame_bottom, text="Save Deraining Image", command=self.save_image,
                                  state=tk.DISABLED, width=32, height=2, font=('Arial', 10, 'bold'))

        self.btn_save.grid(row=2, column=0, columnspan=2, pady=(10, 2), sticky='n')

        btn_frame_bottom.grid_columnconfigure(0, weight=1)
        btn_frame_bottom.grid_columnconfigure(1, weight=1)

        # Bottom middle: Status and guidance area
        guide_frame = tk.Frame(bottom_frame)
        guide_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Label(guide_frame, text="Status", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.help_text = ScrolledText(guide_frame, width=50, height=6, wrap=tk.WORD, state=tk.NORMAL, font=('Arial', 10))
        self.help_text.pack(fill=tk.BOTH, expand=True)
        instructions = """Image Rain Removal System: Remove the rain streaks in the image
        
Operation steps：
1. Click the "Select Image" button and choose an image to be processed (supporting formats such as PNG and JPG).
2. After the image is loaded, the original image will be displayed on the left. At this point, the "Deraining" button becomes available.
3. Click the "Deraining" button, and the program will call the model for processing.
4. After processing is completed, the rain removal result will be displayed on the right side, and the "Save Deraining Image" button will become available.
5. Click the "Save Deraining Image" button to save the rain removal result to your local device.

"""
        self.help_text.insert(tk.END, instructions)
        self.help_text.config(state=tk.DISABLED)

        # Bottom right: Image information area
        info_frame_bottom = tk.LabelFrame(bottom_frame, text="Image information", padx=10, pady=10, font=('Arial', 12, 'bold'))
        info_frame_bottom.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=False)
        self.info_text = tk.Text(info_frame_bottom, height=20, width=35, state=tk.DISABLED, font=('Arial', 10))
        self.info_text.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image with rain streak",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.original_image = Image.open(file_path).convert('RGB')

            self.display_image_on_canvas(self.original_image, self.canvas_original, max_size=512)
            self.update_info_text("Loaded, pending processing")

            self.canvas_derained.delete("all")
            canvas_width = self.canvas_derained.winfo_width()
            canvas_height = self.canvas_derained.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 512
                canvas_height = 512
            self.canvas_derained.create_text(canvas_width // 2, canvas_height // 2, text="Wait for processing...",
                                                 fill='darkgrey', font=('Arial', 16))

            self.derained_image = None
            self.derained_image_tk = None
            self.btn_save.config(state=tk.DISABLED)
            self.btn_process.config(state=tk.NORMAL)

            self.current_file_path = file_path
            self.log_status(f"Loaded image: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Load Error", f"The image cannot be loaded:\n{e}")

    def display_image_on_canvas(self, pil_image, canvas, max_size=512):
        img_width, img_height = pil_image.size
        scale = min(max_size / img_width, max_size / img_height, 1)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)

        canvas.delete("all")

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = new_width
            canvas_height = new_height

        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)

        if canvas == self.canvas_original:
            self.original_image_tk = photo
        else:
            self.derained_image_tk = photo

    def update_info_text(self, status="Processing completed"):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        if self.original_image:
            info = f"Original image information:\n"
            info += f"  File name: {os.path.basename(getattr(self, 'current_file_path', 'N/A'))}\n"
            info += f"  Size: {self.original_image.size[0]} x {self.original_image.size[1]}\n"
            info += f"  Mode: {self.original_image.mode}\n"
            info += "-"*30 + "\n"
        else:
            info = "Original image information: None\n" + "-"*30 + "\n"

        if self.derained_image:
            info += f"Result image information:\n"
            info += f"  Size: {self.derained_image.size[0]} x {self.derained_image.size[1]}\n"
            info += f"  Mode: {self.derained_image.mode}\n"
            info += f"  Status: {status}\n"
        else:
            info += f"Result image information: {status}\n"

        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        self.log_status("Processing...")
        self.btn_process.config(state=tk.DISABLED)
        self.root.update()

        try:
            input_img = self.to_tensor(self.original_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                restored = self.model(input_img)
                if isinstance(restored, (tuple, list)):
                    output_tensor = restored[0]
                else:
                    output_tensor = restored
                output_tensor = torch.clamp(output_tensor, 0, 1)

            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255.0).round().astype(np.uint8)
            self.derained_image = Image.fromarray(output_np)

            self.display_image_on_canvas(self.derained_image, self.canvas_derained, max_size=512)
            self.update_info_text("Deraining completed")
            self.btn_save.config(state=tk.NORMAL)
            self.log_status(f"The rain removal processing of the image has been completed!")

        except Exception as e:
            self.log_status(f"An error occurred during the processing: {e}")
            messagebox.showerror("Process Error", f"Model processing failure:\n{e}")
            self.btn_process.config(state=tk.NORMAL)
            print(f"Detail: {e}")

    def save_image(self):
        if self.derained_image is None:
            return

        default_name = os.path.basename(getattr(self, 'current_file_path', 'derained_image.png'))
        name_without_ext = os.path.splitext(default_name)[0]
        default_filename = f"{name_without_ext}_derained.png"

        file_path = filedialog.asksaveasfilename(
            title="Save the deraining image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=default_filename
        )
        if file_path:
            try:
                self.derained_image.save(file_path)
                self.log_status(f"The image has been saved to: {file_path}")
                messagebox.showinfo("Successful", f"The deraining image has been saved!\nLocation: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the image:\n{e}")

    def log_status(self, message):
        self.help_text.config(state=tk.NORMAL)
        self.help_text.insert(tk.END, f"\n[Status] {message}")
        self.help_text.see(tk.END)
        self.help_text.config(state=tk.DISABLED)
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = DerainGUI(root)
    root.mainloop()