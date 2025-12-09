import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import shutil
import subprocess
import sys
import numpy as np


class GingivitisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Gingivitis App")
        self.root.geometry("700x400")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(False, False)

        self._check_dependencies()

        self._set_icon()

        self.path_var = tk.StringVar()

        self._create_widgets()

    def _check_dependencies(self):
        required_packages = {
            'torch': 'torch',
            'torchvision': 'torchvision',
            'segmentation_models_pytorch': 'segmentation-models-pytorch',
            'albumentations': 'albumentations',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'numpy': 'numpy'
        }

        missing_packages = []
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(pip_name)

        if missing_packages:
            self._show_dependency_install_dialog(missing_packages)

    def _show_dependency_install_dialog(self, missing_packages):
        response = messagebox.askyesno(
            "Missing Dependencies",
            f"The following required packages are missing:\n\n" +
            "\n".join(f"• {pkg}" for pkg in missing_packages) +
            f"\n\nWould you like to install them now?\n\n"
            f"This may take several minutes and requires an internet connection."
        )

        if response:
            self._install_dependencies(missing_packages)
        else:
            messagebox.showwarning(
                "Cannot Continue",
                "The application requires these dependencies to function.\n"
                "The app will now close."
            )
            self.root.destroy()
            sys.exit(1)

    def _install_dependencies(self, packages):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Installing Dependencies")
        progress_window.geometry("500x200")
        progress_window.configure(bg="#f5f5f5")
        progress_window.resizable(False, False)

        progress_window.transient(self.root)
        progress_window.grab_set()

        label = tk.Label(
            progress_window,
            text="Installing required dependencies...\nPlease wait, this may take several minutes.",
            font=("Segoe UI", 11),
            bg="#f5f5f5",
            fg="#2c3e50"
        )
        label.pack(pady=30)

        status_label = tk.Label(
            progress_window,
            text="Starting installation...",
            font=("Segoe UI", 9),
            bg="#f5f5f5",
            fg="#7f8c8d"
        )
        status_label.pack(pady=10)

        def install():
            try:
                for idx, package in enumerate(packages, 1):
                    status_label.config(text=f"Installing {package} ({idx}/{len(packages)})...")
                    progress_window.update()

                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode != 0:
                        raise Exception(f"Failed to install {package}: {result.stderr}")

                status_label.config(text="Installation complete!")
                progress_window.update()

                messagebox.showinfo(
                    "Success",
                    "All dependencies have been installed successfully!\n\n"
                    "Please restart the application."
                )
                progress_window.destroy()
                self.root.destroy()
                sys.exit(0)

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror(
                    "Installation Failed",
                    f"Failed to install dependencies:\n\n{str(e)}\n\n"
                    f"Please install manually using:\n"
                    f"pip install {' '.join(packages)}"
                )
                self.root.destroy()
                sys.exit(1)

        self.root.after(100, install)

    def _set_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
            if os.path.exists(icon_path):
                icon_image = Image.open(icon_path)
                icon_photo = ImageTk.PhotoImage(icon_image)
                self.root.iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Could not load icon: {e}")

    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#f5f5f5")
        main_frame.pack(expand=True, fill="both", padx=40, pady=40)

        title_label = tk.Label(
            main_frame,
            text="Gingivitis Detector! 🦷",
            font=("Segoe UI", 18, "bold"),
            bg="#f5f5f5",
            fg="#2c3e50"
        )
        title_label.pack(pady=(0, 30))

        instruction_frame = tk.Frame(main_frame, bg="white", relief="flat", bd=0)
        instruction_frame.pack(fill="x", pady=(0, 20))

        instruction_label = tk.Label(
            instruction_frame,
            text="Choose the path of the directory with the images :)",
            font=("Segoe UI", 11),
            bg="white",
            fg="#34495e",
            pady=15
        )
        instruction_label.pack()

        path_frame = tk.Frame(main_frame, bg="#f5f5f5")
        path_frame.pack(fill="x", pady=(0, 25))

        self.path_display = tk.Entry(
            path_frame,
            textvariable=self.path_var,
            font=("Segoe UI", 10),
            state='readonly',
            relief="solid",
            bd=1,
            readonlybackground="white",
            fg="#2c3e50"
        )
        self.path_display.pack(fill="x", ipady=8)

        button_frame = tk.Frame(main_frame, bg="#f5f5f5")
        button_frame.pack(pady=(0, 10))

        self.browse_button = tk.Button(
            button_frame,
            text="📂 Browse",
            command=self._browse_directory,
            font=("Segoe UI", 11, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=40,
            pady=12,
            bd=0
        )
        self.browse_button.pack(side="left", padx=5)

        self.submit_button = tk.Button(
            button_frame,
            text="✓ Submit",
            command=self._submit_path,
            font=("Segoe UI", 11, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=40,
            pady=12,
            bd=0
        )
        self.submit_button.pack(side="left", padx=5)

        self._add_hover_effects()

    def _add_hover_effects(self):
        def on_enter_browse(e):
            self.browse_button['bg'] = '#2980b9'

        def on_leave_browse(e):
            self.browse_button['bg'] = '#3498db'

        def on_enter_submit(e):
            self.submit_button['bg'] = '#229954'

        def on_leave_submit(e):
            self.submit_button['bg'] = '#27ae60'

        self.browse_button.bind("<Enter>", on_enter_browse)
        self.browse_button.bind("<Leave>", on_leave_browse)
        self.submit_button.bind("<Enter>", on_enter_submit)
        self.submit_button.bind("<Leave>", on_leave_submit)

    def _browse_directory(self):
        directory = filedialog.askdirectory(
            title="Select Directory with Images"
        )
        if directory:
            self.path_var.set(directory)
            print(f"Directory selected: {directory}")

    def _get_directory_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def _run_model(self, input_dir, weights):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if weights == "Teeth_model_weights.pth":
            temp_dir = os.path.join(script_dir, "temp_teeth_model")
            weights_path = os.path.join(project_root, "weights&results", "Teeth_model_weights.pth")
        else:
            temp_dir = os.path.join(script_dir, "temp_gingivitis_model")
            weights_path = os.path.join(project_root, "weights&results", "Gingivitis_model_weights.pth")

        os.makedirs(temp_dir, exist_ok=True)
        model_script = os.path.join(project_root, "tools", "run_model.py")

        cmd = [
            sys.executable,
            model_script,
            "--weights", weights_path,
            "--input", input_dir,
            "--output", temp_dir
        ]

        try:
            model_name = weights.replace("_weights.pth", "").replace("_model", "")
            print(f"Running {model_name} model: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running teeth extraction: {e.stderr}")
            messagebox.showerror("Extraction Error", f"Failed to run teeth extraction:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            return False

    def _run_get_relevant(self, original_input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, "temp_teeth_model")

        project_root = os.path.dirname(script_dir)
        get_relevant_script = os.path.join(project_root, "tools", "get_relevant.py")

        cmd = [
            sys.executable,
            get_relevant_script,
            original_input_dir,
            temp_dir,
            script_dir
        ]

        try:
            print(f"Running get_relevant: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)

            old_dir = os.path.join(script_dir, "relevant_images")
            new_dir = os.path.join(script_dir, "temp_relevant_images")

            if os.path.exists(old_dir):
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir)
                os.rename(old_dir, new_dir)
                print(f"Renamed 'relevant_images' to 'temp_relevant_images'")

            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running get_relevant: {e.stderr}")
            messagebox.showerror("Get Relevant Error", f"Failed to run get_relevant:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            return False

    def _create_final_results(self, original_input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gingivitis_masks_dir = os.path.join(script_dir, "temp_gingivitis_model")
        final_results_dir = os.path.join(script_dir, "final_result")

        os.makedirs(final_results_dir, exist_ok=True)

        try:
            import cv2

            image_files = [f for f in os.listdir(original_input_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            print(f"Creating final results with green outline for {len(image_files)} images...")

            for img_file in image_files:
                img_path = os.path.join(original_input_dir, img_file)
                mask_name = os.path.splitext(img_file)[0] + ".jpg"
                mask_path = os.path.join(gingivitis_masks_dir, mask_name)

                if not os.path.exists(mask_path):
                    print(f"Warning: No mask found for {img_file}, copying original")
                    img = Image.open(img_path)
                    output_path = os.path.join(final_results_dir, img_file)
                    img.save(output_path)
                    continue

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                if img.size != mask.size:
                    mask = mask.resize(img.size, Image.LANCZOS)

                img_array = np.array(img)
                mask_array = np.array(mask)

                _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                result = img_array.copy()
                cv2.drawContours(result, contours, -1, (0, 255, 0), thickness=3)

                result_img = Image.fromarray(result)
                output_path = os.path.join(final_results_dir, img_file)
                result_img.save(output_path)

            print(f"Final results saved to: {final_results_dir}")
            return True

        except Exception as e:
            print(f"Error creating final results: {str(e)}")
            messagebox.showerror("Final Results Error", f"Failed to create final results:\n{str(e)}")
            return False

    def _cleanup_temp_directories(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dirs = [
            os.path.join(script_dir, "temp_teeth_model"),
            os.path.join(script_dir, "temp_relevant_images"),
            os.path.join(script_dir, "temp_gingivitis_model")
        ]

        try:
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Deleted temporary directory: {temp_dir}")

            print("Cleanup completed successfully")
            return True

        except Exception as e:
            print(f"Warning: Could not clean up some temporary directories: {str(e)}")
            return False

    def _submit_path(self):
        path = self.path_var.get()
        if path:
            try:
                dir_size = self._get_directory_size(path)
                required_space = dir_size * 3

                current_drive = os.path.abspath(__file__).split(os.sep)[0] + os.sep
                free_space = shutil.disk_usage(current_drive).free

                if free_space < required_space:
                    required_gb = required_space / (1024 ** 3)
                    available_gb = free_space / (1024 ** 3)
                    messagebox.showerror(
                        "Insufficient Space",
                        f"Not enough space for the model to work!\n\n"
                        f"Required: {required_gb:.2f} GB\n"
                        f"Available: {available_gb:.2f} GB"
                    )
                else:
                    print(f"Directory path submitted: {path}")

                    if self._run_model(path, "Teeth_model_weights.pth"):
                        if self._run_get_relevant(path):
                            temp_relevant_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "temp_relevant_images")
                            if self._run_model(temp_relevant_dir, "Gingivitis_model_weights.pth"):
                                if self._create_final_results(path):
                                    self._cleanup_temp_directories()
                                    messagebox.showinfo("Success", "Processing completed successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a directory!")


def main():
    root = tk.Tk()
    app = GingivitisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()