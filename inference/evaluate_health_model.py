import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained health model
model = tf.keras.models.load_model("models/health_classifier.h5")

# Test dataset path
test_dir = "datasets/health_dataset/test"

# Image generator
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Predict
pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int)

# Ground truth
true_labels = test_data.classes
class_names = list(test_data.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))
