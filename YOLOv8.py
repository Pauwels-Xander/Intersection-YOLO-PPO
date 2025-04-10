from roboflow import Roboflow
from ultralytics import YOLO
import random, os
import csv


def train_model():
    # Download Roboflow dataset
    rf = Roboflow(api_key="mq7KSElG5TX7TTze2MeF")
    project = rf.workspace("xander-pauwels").project("cars-o1ljf-kbt4r")
    dataset = project.version(1).download("yolov8")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    # Define the path where the best model weights should be saved.
    best_model_path = os.path.join("runs", "train", "cars_trucks_bikes", "weights", "best.pt")

    if os.path.exists(best_model_path):
        print("Loading saved model from:", best_model_path)
        model = YOLO(best_model_path)
    else:
        print("No saved model found. Training a new model...")
        model = YOLO("yolov8n.pt")  # auto‑downloads tiny pretrained weights
        model.train(
            data=data_yaml,
            epochs=30,
            imgsz=640,
            batch=16,
            name="cars_trucks_bikes"
        )
        # Evaluate on validation set
        metrics = model.val(data=data_yaml)
        print("Validation metrics:", metrics)

    return model, dataset


def single_image_inference(model, dataset):
    # Inference on a random test image
    test_images_path = os.path.join(dataset.location, "test", "images")
    test_images = os.listdir(test_images_path)
    img_path = os.path.join(test_images_path, random.choice(test_images))
    results = model.predict(img_path, conf=0.25, save=True)
    results[0].show()  # displays bounding boxes
    print("Single image inference complete ➜ check runs/predict/ for saved output")

def batch_inference_to_csv(model, dataset, output_csv="predictions.csv"):
    # Run inference on all test images and save predictions to a CSV file.
    test_images_path = os.path.join(dataset.location, "test", "images")
    test_images = os.listdir(test_images_path)

    # Open CSV for writing results
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "class", "confidence", "x1", "y1", "x2", "y2"])

        for image_name in test_images:
            img_path = os.path.join(test_images_path, image_name)
            results = model.predict(img_path, conf=0.25, save=False)

            # Loop through predictions in the first result (assuming one image per prediction call)
            for box in results[0].boxes:
                # Get bounding box coordinates and other attributes
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Map numeric class to a label:
                label = "car" if cls == 0 else "truck"  # adjust mapping based on the dataset
                writer.writerow([image_name, label, conf] + coords)

    print(f"Batch inference complete. Predictions saved to {output_csv}")

def main():
    model, dataset = train_model()
    single_image_inference(model, dataset)
    batch_inference_to_csv(model, dataset, output_csv="predictions.csv")
if __name__ == '__main__':
    main()
