import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import time

class DifferenceDetectionModel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("wejden detection model")

        self.frame_top = tk.Frame(self.root)
        self.frame_top.pack()

        self.label_welcome = tk.Label(self.frame_top, text="wejden detection model", font=("Helvetica", 24))
        self.label_welcome.pack(pady=10)

        self.button_load = tk.Button(self.frame_top, text="Load Images", command=self.load_images)
        self.button_load.pack()

        self.button_compare = tk.Button(self.frame_top, text="Start the model", command=self.compare_and_annotate)
        self.button_compare.pack()

        self.button_save = tk.Button(self.frame_top, text="Save Detection in JSON", command=self.save_to_json)
        self.button_save.pack()

        self.frame_images = tk.Frame(self.root)
        self.frame_images.pack()

        self.label_original = tk.Label(self.frame_images)
        self.label_original.grid(row=0, column=0, padx=10)
        self.label_error = tk.Label(self.frame_images)
        self.label_error.grid(row=0, column=1, padx=10)

        self.frame_errors = tk.Frame(self.root)
        self.frame_errors.pack(fill=tk.BOTH, expand=True, pady=10)

        self.label_error_types = tk.Label(self.frame_errors, text="Types d'erreurs détectés:", font=("Helvetica", 16))
        self.label_error_types.pack()
        self.text_errors = tk.Text(self.frame_errors, height=10, font=("Helvetica", 14))
        self.text_errors.pack(fill=tk.BOTH, expand=True)

        self.img1 = None
        self.img2 = None
        self.path1 = None
        self.path2 = None

        self.results = []  # List to store detection results

    def load_images(self):
        self.path1 = filedialog.askopenfilename()
        if not self.path1:
            return
        self.img1 = Image.open(self.path1)
        self.img1.thumbnail((250, 250))  # Resize for display
        img1_display = ImageTk.PhotoImage(self.img1)

        self.path2 = filedialog.askopenfilename()
        if not self.path2:
            return
        self.img2 = Image.open(self.path2)
        self.img2.thumbnail((250, 250))  # Resize for display
        img2_display = ImageTk.PhotoImage(self.img2)

        self.label_original.imgtk = img1_display
        self.label_original.configure(image=img1_display)
        self.label_error.imgtk = img2_display
        self.label_error.configure(image=img2_display)

    def compare_and_annotate(self):
        if self.img1 is None or self.img2 is None:
            return

        original = cv2.cvtColor(np.array(self.img1), cv2.COLOR_RGB2GRAY)
        error = cv2.cvtColor(np.array(self.img2), cv2.COLOR_RGB2GRAY)

        min_height = min(original.shape[0], error.shape[0])
        min_width = min(original.shape[1], error.shape[1])
        original = cv2.resize(original, (min_width, min_height))
        error = cv2.resize(error, (min_width, min_height))

        difference = cv2.absdiff(original, error)
        thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]

        # Prepare the image where we'll draw the differences
        marked = cv2.cvtColor(error.copy(), cv2.COLOR_GRAY2BGR)

        # Find the contours of the differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        error_types = []  # To store the types of errors

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 255, 0), 1)
            error_type = self.classify_error(w*h)
            error_types.append(f"{error_type} detected at coordinates: ({x}, {y}), Size: ({w}x{h})")

        # Convert marked image for display
        marked_display = Image.fromarray(marked)
        marked_display.thumbnail((250, 250))  # Resize for display
        marked_display = ImageTk.PhotoImage(marked_display)

        # Display the marked image
        self.label_error.imgtk = marked_display
        self.label_error.configure(image=marked_display)

        # Update the text box with the error types
        self.text_errors.delete('1.0', tk.END)
        self.text_errors.insert(tk.END, "\n".join(error_types) if error_types else "No significant errors detected.")

        # Store the detection results
        timestamp = int(time.time() * 1000)  # Get current time in milliseconds
        result = {
            "timestamp": timestamp,
            "errors": error_types
        }
        self.results.append(result)

    def classify_error(self, area):
        if area < 50:
            return "Minor Error"
        elif area < 200:
            return "Moderate Error"
        else:
            return "Major Error"

    def save_to_json(self):
        if not self.results:
            return

        filename = f"detection_{int(time.time() * 1000)}.json"
        with open(filename, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

        print(f"Detection results saved to {filename}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model = DifferenceDetectionModel()
    model.run()

       
