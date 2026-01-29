import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/growth_stage_best.h5"
IMG_SIZE = (224, 224)
OUTPUT_DIR = "gradcam_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Identify last conv layer
# -----------------------------
LAST_CONV_LAYER = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        LAST_CONV_LAYER = layer.name
        break

if LAST_CONV_LAYER is None:
    raise ValueError("No Conv2D layer found for Grad-CAM")

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Grad-CAM Function
# -----------------------------
def generate_gradcam(image_path):
    img_tensor = preprocess_image(image_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, IMG_SIZE)

    # Overlay on original image
    original = cv2.imread(image_path)
    original = cv2.resize(original, IMG_SIZE)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    output_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_" + os.path.basename(image_path)
    )

    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM saved to: {output_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gradcam_growth.py <image_path>")
        sys.exit(1)

    generate_gradcam(sys.argv[1])
