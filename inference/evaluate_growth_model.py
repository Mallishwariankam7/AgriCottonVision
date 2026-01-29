import tensorflow as tf
import numpy as np
import cv2
import os


# Load trained growth model

MODEL_PATH = "models/growth_stage_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)


# Class index mapping
# (matches folder order)

CLASS_NAMES = [
    "Phase 1: Budding",
    "Phase 2: Flowering",
    "Phase 3: Bursting",
    "Phase 4: Harvest Ready"
]


# Image preprocessing function

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Predict growth stage

def predict_growth_stage(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)

    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    return {
        "stage": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":
    # change path to any test image
    test_image_path = "datasets/growth_stage/phase4_harvest/p4_15.jpeg"
    

    result = predict_growth_stage(test_image_path)

    print("\nGrowth Stage Prediction")
    print("-----------------------")
    print(f"Predicted Stage : {result['stage']}")
    print(f"Confidence      : {result['confidence']}%")
