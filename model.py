import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.optimizers import Adam
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, precision_recall_curve)
import tensorflow as tf
from keras.applications import MobileNetV2
from tensorflow.keras.metrics import Precision, Recall, AUC
import seaborn as sns
import gc
import shutil


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
data_dir = 'Skin_Data'
img_size = (224, 224)
batch_size = 16
n_folds = 5

def load_images(folder_path, expected_count):
    images = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    assert len(images) == expected_count, f"Expected {expected_count} images, got {len(images)}"
    return np.array(images)

def build_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    base_model.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )
    return model

# Load and validate data
print("Loading datasets...")
cancer_train = load_images(os.path.join(data_dir, 'Cancer/Training'), 50)
cancer_test = load_images(os.path.join(data_dir, 'Cancer/Testing'), 50)
non_cancer_train = load_images(os.path.join(data_dir, 'Non_Cancer/Training'), 50)
non_cancer_test = load_images(os.path.join(data_dir, 'Non_Cancer/Testing'), 50)

# Prepare datasets
x_train_full = np.concatenate((cancer_train, non_cancer_train))
y_train_full = np.concatenate(([1]*50, [0]*50))
x_test = np.concatenate((cancer_test, non_cancer_test))
y_test = np.concatenate(([1]*50, [0]*50))

# Initialize cross-validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_models = []
history_list = []

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Cross-validation training
for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_full, y_train_full)):
    print(f"\n=== Training Fold {fold+1}/{n_folds} ===")
    
    model = build_model()
    
    callbacks = [
        ModelCheckpoint(f'best_fold{fold+1}.h5', save_best_only=True,
                       monitor='val_recall', mode='max'),
        EarlyStopping(monitor='val_recall', patience=12, mode='max',
                     restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=6, mode='max')
    ]
    
    history = model.fit(
        train_datagen.flow(x_train_full[train_idx], y_train_full[train_idx], batch_size=batch_size),
        validation_data=val_test_datagen.flow(x_train_full[val_idx], y_train_full[val_idx]),
        epochs=60,
        class_weight={0: 1, 1: 1.8},
        callbacks=callbacks,
        verbose=1
    )
    
    fold_models.append(model)
    history_list.append(history)
    gc.collect()

# Threshold optimization
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    return thresholds[np.argmax(f1_scores)]

# Generate predictions
test_probas = np.zeros((len(x_test), n_folds))
val_probas = np.zeros((len(x_train_full), n_folds))

for fold in range(n_folds):
    model = tf.keras.models.load_model(f'best_fold{fold+1}.h5')
    
    # Rename layers to avoid conflicts
    for layer in model.layers:
        layer._name = f"fold{fold+1}_{layer.name}"
    
    val_preds = model.predict(val_test_datagen.flow(x_train_full, shuffle=False))
    val_probas[:, fold] = val_preds.flatten()
    
    test_preds = model.predict(val_test_datagen.flow(x_test, shuffle=False))
    test_probas[:, fold] = test_preds.flatten()

val_probas_avg = np.mean(val_probas, axis=1)
optimal_threshold = find_optimal_threshold(y_train_full, val_probas_avg)
print(f"\nOptimal threshold: {optimal_threshold:.4f}")

test_probas_avg = np.mean(test_probas, axis=1)
y_pred_classes = (test_probas_avg > optimal_threshold).astype(int)

# Evaluation
print("\n=== Final Evaluation ===")
print(classification_report(y_test, y_pred_classes, target_names=['Non-Cancer', 'Cancer']))
cm = confusion_matrix(y_test, y_pred_classes)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"\nTotal Misclassified: {fp + fn}/{len(y_test)} ({(fp + fn)/len(y_test):.2%})")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\nAccuracy: {accuracy:.1f}")


# ==== Final Working Model Saving ====
def create_ensemble(models):
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    outputs = [model(input_layer) for model in models]
    avg_output = tf.keras.layers.Average()(outputs)
    return tf.keras.Model(inputs=input_layer, outputs=avg_output)

ensemble_model = create_ensemble(fold_models)

# 2. HDF5 format
export_path = "skin_cancer_model.h5"
ensemble_model.save(export_path)
print(f"\n✅ Model saved as {export_path}")

