import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import shutil
import subprocess
import sys


class GingivitisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Gingivitis App")
        self.root.geometry("700x400")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(False, False)

        self._set_icon()

        self.path_var = tk.StringVar()

        self._create_widgets()

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

    def _run_teeth_extraction(self, input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, "temp_teeth_model")
        os.makedirs(temp_dir, exist_ok=True)

        project_root = os.path.dirname(script_dir)
        weights_path = os.path.join(project_root, "weights&results", "Teeth_model_weights.pth")
        teeth_extraction_script = os.path.join(project_root, "tools", "teeth_extraction.py")

        cmd = [
            sys.executable,
            teeth_extraction_script,
            "--weights", weights_path,
            "--input", input_dir,
            "--output", temp_dir
        ]

        try:
            print(f"Running teeth extraction: {' '.join(cmd)}")
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

                    if self._run_teeth_extraction(path):
                        print("Teeth extraction completed, now running get_relevant...")
                        if self._run_get_relevant(path):
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