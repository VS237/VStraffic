import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

class TrafficSignClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("800x600")
        
        # Load model and class names
        self.model = load_model('best_model.h5')
        self.class_names = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
            'No passing', 'No passing for vehicles over 3.5 tons'
        ]  # Update with your actual class names

        self.create_widgets()

    def create_widgets(self):
        # GUI elements (same as before)
        self.image_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.upload_btn = tk.Button(
            self.root,
            text="Upload Image",
            command=self.upload_image,
            font=('Arial', 12),
            bg='#4CAF50',
            fg='white'
        )
        self.upload_btn.pack(pady=10)

        self.prediction_label = tk.Label(
            self.root,
            text="Prediction will appear here",
            font=('Arial', 14),
            fg='blue'
        )
        self.prediction_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                self.display_image(file_path)
                class_id, class_name, confidence = self.predict_traffic_sign(file_path)
                self.prediction_label.config(
                    text=f"Prediction: {class_name} (Class {class_id})\nConfidence: {confidence:.2%}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((600, 400))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def predict_traffic_sign(self, image_path):
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
            
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30))  # Model expects 30x30
        
        # Normalize and add batch dimension
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img, verbose=0)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)
        
        return class_id, self.class_names[class_id], confidence

# Standalone function (can be imported by other scripts)
def predict_from_image(model_path, class_names, image_path):
    
    model = load_model(model_path)
    classifier = TrafficSignClassifier(tk.Tk())  # Temporary window
    classifier.model = model
    classifier.class_names = class_names
    return classifier.predict_traffic_sign(image_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignClassifier(root)
    root.mainloop()