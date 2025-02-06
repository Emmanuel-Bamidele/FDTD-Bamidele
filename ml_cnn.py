import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS AND DIRECTORY PATHS
# =============================================================================

# Image dimensions (adjust as needed)
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3

# Training parameters
BATCH_SIZE = 8
EPOCHS = 100  # adjust based on your dataset size and training behavior

# Directories
IMAGE_DIR = 'CNN_data/maps'          # Folder containing the spatial map images (one set)
MODEL_SAVE_DIR = 'CNN_data/models_cnn'
PREDICTIONS_DIR = 'CNN_data/predicted_maps'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# CSV mapping file (this CSV maps each image filename to its simulation parameters)
CSV_MAPPING_PATH = 'CNN_data/mapping.csv'

# =============================================================================
# 2. LOAD CSV MAPPING AND PREPARE THE DATASET
# =============================================================================

# The CSV should have columns: 
# "image", "thickness", "wavelength", "material", "absorbed_power", "absorbed_flux"
df = pd.read_csv(CSV_MAPPING_PATH)

# Create full image paths
df['image_path'] = df['image'].apply(lambda x: os.path.join(IMAGE_DIR, x))

# One-hot encode the "material" column.
# This will create new columns, e.g. "material_Au" and "material_Ag".
material_dummies = pd.get_dummies(df['material'], prefix='material')
df = pd.concat([df, material_dummies], axis=1)

# Build the input features:
# We'll use thickness, wavelength, and the one-hot encoded material columns.
input_feature_columns = ['thickness', 'wavelength', 'material_Au', 'material_Ag']
X_data = df[input_feature_columns].values.astype(np.float32)

# For the target, load the corresponding spatial map images.
def load_and_preprocess_image(path):
    """
    Reads an image from disk, resizes it to the specified dimensions,
    and normalizes pixel values to the [0,1] range.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Load all images into a TensorFlow tensor.
Y_data_list = [load_and_preprocess_image(path) for path in df['image_path']]
Y_data = tf.stack(Y_data_list)  # shape: (num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

print(f"Loaded {len(X_data)} samples.")

# Optionally, normalize X_data using a scaler.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_data_norm = scaler.fit_transform(X_data)

# =============================================================================
# 3. BUILD A MODEL THAT MAPS PARAMETER VECTORS TO SPATIAL MAPS
# =============================================================================

def build_parameter_to_image_model(input_dim, output_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    """
    Builds a model that takes a vector of simulation parameters (e.g., thickness, wavelength, material)
    and produces an image (the spatial map) as output.
    
    The architecture uses dense layers to expand the input to a latent space,
    then reshapes it and applies Conv2DTranspose layers to generate an image.
    """
    inputs = keras.Input(shape=(input_dim,))
    # Expand the input vector into a larger latent vector.
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(8 * 8 * 128, activation='relu')(x)  # create a small "image" of shape (8, 8, 128)
    x = layers.Reshape((8, 8, 128))(x)
    
    # Upsample the latent image step by step.
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 16x16
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)   # 32x32
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)   # 64x64
    x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(x)   # 128x128
    x = layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding='same', activation='relu')(x)    # 256x256
    outputs = layers.Conv2D(CHANNELS, kernel_size=3, activation='sigmoid', padding='same')(x)
    
    model = keras.Model(inputs, outputs)
    return model

input_dim = X_data_norm.shape[1]  # now 4 (thickness, wavelength, material_Au, material_Ag)
model = build_parameter_to_image_model(input_dim=input_dim,
                                         output_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =============================================================================
# 4. TRAIN THE MODEL
# =============================================================================

# Create a tf.data.Dataset from the normalized inputs and images.
dataset = tf.data.Dataset.from_tensor_slices((X_data_norm, Y_data))
dataset = dataset.shuffle(buffer_size=len(X_data)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Optionally, split into training and validation sets (e.g., 80/20 split)
num_samples = len(X_data)
train_size = int(0.8 * num_samples)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Save the trained model (using the native Keras format)
model_save_path = os.path.join(MODEL_SAVE_DIR, 'param_to_image_model.keras')
model.save(model_save_path)
print("Model saved at:", model_save_path)

# =============================================================================
# 5. PREDICTION, COMPARISON, & ORGANIZATION OF OUTPUTS
# =============================================================================

def compute_metrics(image_array):
    """
    Given an image (as a NumPy array with values in [0,1]), compute two example metrics.
    Here, we use the mean pixel value as a proxy for absorbed power and
    the standard deviation as a proxy for absorbed flux.
    (Adjust these computations as needed.)
    """
    abs_power = image_array.mean()
    abs_flux = image_array.std()
    return abs_power, abs_flux

# Prepare lists for additional plots.
actual_ap_list = []
predicted_ap_list = []
actual_af_list = []
predicted_af_list = []

# Loop over each sample from the CSV.
for idx, row in df.iterrows():
    # Prepare the input vector (thickness, wavelength, material_Au, material_Ag)
    input_vector = np.array([[row['thickness'], row['wavelength'], 
                              row.get('material_Au', 0), row.get('material_Ag', 0)]], dtype=np.float32)
    # Normalize using the same scaler
    input_vector_norm = scaler.transform(input_vector)
    
    # Predict the spatial map using the trained model
    pred_img = model.predict(input_vector_norm)[0]  # shape: (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    
    # Compute predicted metrics (proxies for absorbed power and absorbed flux)
    pred_abs_power, pred_abs_flux = compute_metrics(pred_img)
    
    # For additional plots, use the actual values from CSV.
    actual_ap = row['absorbed_power']
    actual_af = row['absorbed_flux']
    actual_ap_list.append(actual_ap)
    predicted_ap_list.append(pred_abs_power)
    actual_af_list.append(actual_af)
    predicted_af_list.append(pred_abs_flux)
    
    # Convert predicted image from float [0,1] to uint8 [0,255]
    pred_img_uint8 = (pred_img * 255).astype(np.uint8)
    # Convert from RGB to BGR (for OpenCV)
    pred_img_bgr = cv2.cvtColor(pred_img_uint8, cv2.COLOR_RGB2BGR)
    
    # Load the actual (target) image from disk for comparison.
    actual_img_tensor = load_and_preprocess_image(row['image_path'])
    actual_img_uint8 = (actual_img_tensor.numpy() * 255).astype(np.uint8)
    actual_img_bgr = cv2.cvtColor(actual_img_uint8, cv2.COLOR_RGB2BGR)
    
    # Create a combined canvas of the actual and predicted images (side-by-side).
    combined_images = cv2.hconcat([actual_img_bgr, pred_img_bgr])
    
    # Create a header canvas (white background) to hold the titles and parameters.
    header_height = 60
    header_width = combined_images.shape[1]
    header = np.ones((header_height, header_width, 3), dtype=np.uint8) * 255  # white background
    
    # Add the titles.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness_text = 2
    # "Actual" on the left; "Predicted" on the right.
    cv2.putText(header, "Actual", (20, 25), font, font_scale, (0, 0, 0), thickness_text, cv2.LINE_AA)
    cv2.putText(header, "Predicted", (header_width//2 + 20, 25), font, font_scale, (0, 0, 0), thickness_text, cv2.LINE_AA)
    
    # Add the second line with parameters and metrics.
    params_text = f"T:{row['thickness']}, W:{row['wavelength']}, M:{row['material']}"
    cv2.putText(header, params_text, (20, 55), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Combine header and images vertically.
    final_canvas = cv2.vconcat([header, combined_images])
    
    # Build a filename that encodes the parameters and computed metrics.
    filename = f"T{row['thickness']}_W{row['wavelength']}_M{row['material']}_AP{pred_abs_power:.2f}_AF{pred_abs_flux:.2f}_{row['image']}"
    save_path = os.path.join(PREDICTIONS_DIR, filename)
    
    # Save the final canvas.
    cv2.imwrite(save_path, final_canvas)
    print(f"Saved combined image for {row['image']} as: {save_path}")

print("All predicted comparison images have been saved in:", PREDICTIONS_DIR)

# =============================================================================
# 6. ADDITIONAL PLOTS FOR PUBLICATION
# =============================================================================

# Scatter plot: Actual vs. Predicted Absorbed Power
plt.figure(figsize=(8, 6))
plt.scatter(actual_ap_list, predicted_ap_list, color='blue', alpha=0.7, label='Data points')
plt.plot([min(actual_ap_list), max(actual_ap_list)], [min(actual_ap_list), max(actual_ap_list)], 'r--', label='Ideal')
plt.xlabel("Actual Absorbed Power")
plt.ylabel("Predicted Absorbed Power")
plt.title("Actual vs. Predicted Absorbed Power")
plt.legend()
power_scatter_path = os.path.join(PREDICTIONS_DIR, "scatter_absorbed_power.png")
plt.savefig(power_scatter_path)
plt.close()
print(f"Saved scatter plot for absorbed power at: {power_scatter_path}")

# Scatter plot: Actual vs. Predicted Absorbed Flux
plt.figure(figsize=(8, 6))
plt.scatter(actual_af_list, predicted_af_list, color='green', alpha=0.7, label='Data points')
plt.plot([min(actual_af_list), max(actual_af_list)], [min(actual_af_list), max(actual_af_list)], 'r--', label='Ideal')
plt.xlabel("Actual Absorbed Flux")
plt.ylabel("Predicted Absorbed Flux")
plt.title("Actual vs. Predicted Absorbed Flux")
plt.legend()
flux_scatter_path = os.path.join(PREDICTIONS_DIR, "scatter_absorbed_flux.png")
plt.savefig(flux_scatter_path)
plt.close()
print(f"Saved scatter plot for absorbed flux at: {flux_scatter_path}")

# Training History Plot (Loss vs. Epochs)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
# Only plot validation loss if it exists in the history.
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
history_plot_path = os.path.join(PREDICTIONS_DIR, "training_history.png")
plt.savefig(history_plot_path)
plt.close()
print(f"Saved training history plot at: {history_plot_path}")
