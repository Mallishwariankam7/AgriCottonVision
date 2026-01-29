import tensorflow as tf
import numpy as np
import cv2
import sys
import json
import os

# Model Paths (FINAL)

GROWTH_MODEL_PATH = "models/growth_stage_best.h5"
HEALTH_MODEL_PATH = "models/health_classifier.h5"

IMG_SIZE = (224, 224)


# Class Labels
GROWTH_CLASSES = [
    "Phase 1: Vegetative/Budding",
    "Phase 2: Flowering",
    "Phase 3: Bursting",
    "Phase 4: Harvest Ready"
]


# Load Models
growth_model = tf.keras.models.load_model(GROWTH_MODEL_PATH)
health_model = tf.keras.models.load_model(HEALTH_MODEL_PATH)

# Image Preprocessing

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Inference Function

def evaluate_cotton_crop(image_path):
    img = preprocess_image(image_path)


    # Growth Stage Prediction

    growth_preds = growth_model.predict(img, verbose=0)[0]
    growth_index = int(np.argmax(growth_preds))
    growth_stage = GROWTH_CLASSES[growth_index]


    # Health Prediction
    # Model outputs probability of DISEASE

    disease_prob = float(health_model.predict(img, verbose=0)[0][0])
    healthy_prob = 1.0 - disease_prob

    health_status = "diseased" if disease_prob >= 0.5 else "healthy"
    health_score = healthy_prob * 100  # 0–100, higher = healthier


    # Final Output

    result = {
        "image": image_path,
        "growth_stage": growth_stage,
        "is_ripped": True if growth_index >= 2 else False,
        "health_status": health_status,
        "health_score": round(health_score, 2)
    }

    return result


# Main Runner

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_cotton_crop.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output = evaluate_cotton_crop(image_path)
    print(json.dumps(output, indent=4))
