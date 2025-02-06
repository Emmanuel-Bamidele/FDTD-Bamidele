import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  # For image annotation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap  # <-- New: import SHAP

# ----------------------------
# 1. Load and Preprocess Simulation Data
# ----------------------------

DATA_PATH = 'training_data.csv'
PLOTS_DIR = 'plots_01_02'           # Folder where simulation plots (e.g., power_density_map_*.png) are stored
LABELED_PLOTS_DIR = 'labeled_plots'   # Where annotated images will be saved
PAPER_PLOTS_DIR = 'MLP_data/paper_plots'       # Where additional plots (for your paper) are stored

# Create directories if they don't exist
for directory in [LABELED_PLOTS_DIR, PAPER_PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Data file '{DATA_PATH}' not found.")

# Load the CSV data
df = pd.read_csv(DATA_PATH)

# Drop any columns starting with "pixel_" (these are high-dimensional and not needed now)
df = df[[col for col in df.columns if not col.startswith('pixel_')]]

# We assume the CSV has at least the following columns:
#   - Input features: "wavelength", "layer_0_thickness", "layer_1_thickness", "layer_2_thickness"
#   - Material columns: "layer_0_material", "layer_1_material", "layer_2_material"
#   - Targets: "absorbed_power", "absorbed_flux"
target_cols = ['absorbed_power', 'absorbed_flux']

# Separate features and targets
features_df = df.drop(columns=target_cols)
targets_df = df[target_cols]

feature_columns = features_df.columns.tolist()

# Identify which columns are categorical (e.g., materials) and which are numeric.
material_cols = [col for col in features_df.columns if 'material' in col]
numeric_cols = [col for col in features_df.columns if col not in material_cols]

# Ensure numeric columns are numeric and fill missing values with the mean.
for col in numeric_cols:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

# One-hot encode the material columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_material = encoder.fit_transform(features_df[material_cols])
encoded_material_df = pd.DataFrame(encoded_material, 
                                   columns=encoder.get_feature_names_out(material_cols))
# Drop the original material columns and concatenate the one-hot encoded ones
features_df = features_df.drop(columns=material_cols)
features_df = pd.concat([features_df.reset_index(drop=True), encoded_material_df.reset_index(drop=True)], axis=1)

# Convert features and targets to NumPy arrays
X = features_df.values
y = targets_df.values  # Shape: (num_samples, 2)

# ----------------------------
# 2. Split and Scale the Data
# ----------------------------

# (We use a train-test split for model training; later we predict for all samples.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ----------------------------
# 3. Build and Train an Improved Regression Model (MLP)
# ----------------------------

input_dim = X_train_scaled.shape[1]

# Improved model: increased neurons, extra layers, and dropout for regularization.
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(2, activation='linear')  # 2 outputs: absorbed power and absorbed flux
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse',
              metrics=['mae'])

model.summary()

# Increased epochs and early stopping patience may help; adjust these as needed.
history = model.fit(X_train_scaled, y_train_scaled, 
                    epochs=150,
                    batch_size=16,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
                    verbose=1)

# ----------------------------
# 4. Evaluate the Model
# ----------------------------

test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Test MAE (scaled): {test_mae}")

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


# ----------------------------
# 5. SHAP Analysis
# ----------------------------

# 1) Ensure your feature_columns match the final DataFrame
feature_columns = features_df.columns.tolist()
print("Number of features in DataFrame:", len(feature_columns))
# Should match X.shape[1]

# 2) Create single-output Keras model for "absorbed power"
custom_input = tf.keras.Input(shape=(input_dim,))
full_output = model(custom_input)      # shape (None, 2)
power_output = full_output[:, 0:1]     # shape (None, 1)
wrapped_model = tf.keras.Model(inputs=custom_input, outputs=power_output)
_ = wrapped_model(X_train_scaled[:10]) # build the model

# 3) Set up KernelExplainer
explainer = shap.KernelExplainer(
    model=wrapped_model.predict, 
    data=X_train_scaled[:50]  # background subset for SHAP baseline
)

# 4) Choose some samples to explain
samples_to_explain = X_train_scaled[:10]  # shape (10, number_of_features)

# 5) Compute SHAP values
shap_values = explainer.shap_values(samples_to_explain, nsamples=100)

# 6) Debug: Check shapes
print("Raw shap_values type/shape:", type(shap_values), np.array(shap_values).shape)
print("samples_to_explain shape:", samples_to_explain.shape)
print("len(feature_columns):", len(feature_columns))

# If shap_values is a single-element list, unwrap it
if isinstance(shap_values, list) and len(shap_values) == 1:
    shap_values = shap_values[0]  # e.g. (1, 10, features) or (10, features)

# If it's (1, num_samples, num_features), unwrap that first dimension
if shap_values.ndim == 3 and shap_values.shape[0] == 1:
    shap_values = shap_values[0]
    
# If it's (num_samples, 1, num_features), unwrap the second dimension
if shap_values.ndim == 3 and shap_values.shape[1] == 1:
    shap_values = shap_values[:, 0, :]

# If shap_values is (num_samples, num_features, 1), unwrap the last dimension:
if shap_values.ndim == 3 and shap_values.shape[2] == 1:
    shap_values = shap_values[:,:,0]
else:
    # Automatic removal of all 1-sized dimensions
    shap_values = np.squeeze(shap_values)

# STEP A: Which features do you want to keep?
features_to_keep = [
    "wavelength",
    "layer_1_material_Au",
    "layer_1_material_Ag",
    "layer_1_thickness"
]

# STEP B: Create a dictionary mapping old names to new, more readable labels
rename_map = {
    "wavelength": "Wavelength",
    "layer_1_material_Au": "Au",
    "layer_1_material_Ag": "Ag",
    "layer_1_thickness": "Thickness"
}

# STEP C: Find the column indices corresponding to the features you want to keep
indices_to_keep = [feature_columns.index(f) for f in features_to_keep]

# STEP D: Subset your shap_values and samples_to_explain
shap_values_subset = shap_values[:, indices_to_keep]       # (num_samples, 4)
X_subset = samples_to_explain[:, indices_to_keep]          # (num_samples, 4)

# STEP E: Get the new, human‐readable names for just the kept features
kept_feature_names = [rename_map[f] for f in features_to_keep]


# Or simply squeeze any extra 1-dims if you’re sure you only have one output:
# shap_values = np.squeeze(shap_values)

print("Final shap_values shape:", shap_values.shape)

# 7) Plot
shap.summary_plot(
    shap_values_subset,
    X_subset,
    feature_names=kept_feature_names,
    show=False
)
plt.title("SHAP Summary Plot (KernelExplainer) for Absorbed Power")
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()


# ----------------------------
# 6. Annotate Simulation Plots with Actual and Predicted Values
# ----------------------------

def annotate_image_with_both(image_path, true_power, true_flux, pred_power, pred_flux, output_path):
    """
    Loads an image (or creates a blank one if not found), overlays text with both the true
    and predicted absorbed power and flux values, and saves the annotated image.
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create a blank white image if the file is not found
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Prepare text labels: using different colors for true (red) and predicted (blue)
    text_true_power = f"True Absorbed Power: {true_power:.2f}"
    text_pred_power = f"Predicted Absorbed Power: {pred_power:.2f}"
    text_true_flux = f"True Absorbed Flux: {true_flux:.2f}"
    text_pred_flux = f"Predicted Absorbed Flux: {pred_flux:.2f}"
    cv2.putText(img, text_true_power, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, text_pred_power, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, text_true_flux, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, text_pred_flux, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(output_path, img)

# Compute predictions for the entire dataset
X_all_scaled = scaler_X.transform(features_df.values)
y_all_pred_scaled = model.predict(X_all_scaled)
y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled)

# Annotate images for each sample.
# Note: The loop now starts at 1 to match the data indexing.
for i in range(1, len(df)+1):
    image_filename = f"power_density_map_{i}.png"  # Adjust naming if needed.
    image_path = os.path.join(PLOTS_DIR, image_filename)
    output_filename = f"annotated_{i}.png"
    output_path = os.path.join(LABELED_PLOTS_DIR, output_filename)
    # Since our arrays are 0-indexed, we use i-1.
    true_power = y[i-1, 0]
    true_flux = y[i-1, 1]
    pred_power = y_all_pred[i-1, 0]
    pred_flux = y_all_pred[i-1, 1]
    annotate_image_with_both(image_path, true_power, true_flux, pred_power, pred_flux, output_path)

print("Annotated images with actual and predicted values saved in:", LABELED_PLOTS_DIR)

# ----------------------------
# 7. Generate Additional Plots for the Paper
# ----------------------------

# 7a. Scatter Plot: Actual vs. Predicted for Absorbed Power
plt.figure(figsize=(8,6))
plt.scatter(y[:,0], y_all_pred[:,0], color='blue', alpha=0.6, label='Data points')
plt.plot([y[:,0].min(), y[:,0].max()], [y[:,0].min(), y[:,0].max()], 'r--', label='Ideal')
plt.xlabel("Actual Absorbed Power")
plt.ylabel("Predicted Absorbed Power")
plt.title("Actual vs. Predicted Absorbed Power")
plt.legend()
plt.grid(True)
power_scatter_path = os.path.join(PAPER_PLOTS_DIR, "actual_vs_predicted_absorbed_power.png")
plt.savefig(power_scatter_path)
plt.close()

# 7b. Scatter Plot: Actual vs. Predicted for Absorbed Flux
plt.figure(figsize=(8,6))
plt.scatter(y[:,1], y_all_pred[:,1], color='green', alpha=0.6, label='Data points')
plt.plot([y[:,1].min(), y[:,1].max()], [y[:,1].min(), y[:,1].max()], 'r--', label='Ideal')
plt.xlabel("Actual Absorbed Flux")
plt.ylabel("Predicted Absorbed Flux")
plt.title("Actual vs. Predicted Absorbed Flux")
plt.legend()
plt.grid(True)
flux_scatter_path = os.path.join(PAPER_PLOTS_DIR, "actual_vs_predicted_absorbed_flux.png")
plt.savefig(flux_scatter_path)
plt.close()

# 7c. Combined Scatter Plot for Both Targets (using subplots)
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(y[:,0], y_all_pred[:,0], color='blue', alpha=0.6)
axs[0].plot([y[:,0].min(), y[:,0].max()], [y[:,0].min(), y[:,0].max()], 'r--')
axs[0].set_xlabel("Actual Absorbed Power")
axs[0].set_ylabel("Predicted Absorbed Power")
axs[0].set_title("Absorbed Power")
axs[0].grid(True)

axs[1].scatter(y[:,1], y_all_pred[:,1], color='green', alpha=0.6)
axs[1].plot([y[:,1].min(), y[:,1].max()], [y[:,1].min(), y[:,1].max()], 'r--')
axs[1].set_xlabel("Actual Absorbed Flux")
axs[1].set_ylabel("Predicted Absorbed Flux")
axs[1].set_title("Absorbed Flux")
axs[1].grid(True)

plt.suptitle("Actual vs. Predicted Values for Absorbed Power and Flux")
combined_scatter_path = os.path.join(PAPER_PLOTS_DIR, "actual_vs_predicted_combined.png")
plt.savefig(combined_scatter_path)
plt.close()

# 7d. Residual Plots (Histogram of errors for both targets)
residuals_power = y[:,0] - y_all_pred[:,0]
residuals_flux = y[:,1] - y_all_pred[:,1]

plt.figure(figsize=(8,6))
plt.hist(residuals_power, bins=30, color='blue', alpha=0.7)
plt.xlabel("Residuals (Actual - Predicted) Absorbed Power")
plt.ylabel("Frequency")
plt.title("Residuals Histogram for Absorbed Power")
residuals_power_path = os.path.join(PAPER_PLOTS_DIR, "residuals_absorbed_power.png")
plt.savefig(residuals_power_path)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(residuals_flux, bins=30, color='green', alpha=0.7)
plt.xlabel("Residuals (Actual - Predicted) Absorbed Flux")
plt.ylabel("Frequency")
plt.title("Residuals Histogram for Absorbed Flux")
residuals_flux_path = os.path.join(PAPER_PLOTS_DIR, "residuals_absorbed_flux.png")
plt.savefig(residuals_flux_path)
plt.close()

# 7e. Training History Plot
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
history_plot_path = os.path.join(PAPER_PLOTS_DIR, "training_history_loss.png")
plt.savefig(history_plot_path)
plt.close()

print("Additional plots for the paper saved in:", PAPER_PLOTS_DIR)

# ----------------------------
# 8. Save the Regression Model and Training History
# ----------------------------

MODEL_SAVE_DIR = os.path.join("models_01_02")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# Save in the native Keras format (avoids legacy HDF5 warnings)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "regression_model_for_absorption.keras")
model.save(MODEL_SAVE_PATH)
print(f"Trained regression model saved as '{MODEL_SAVE_PATH}'.")
