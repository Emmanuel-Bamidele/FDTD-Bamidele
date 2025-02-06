import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import SiO2, Si3N4, ITO, Au, Ag  # Import available materials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import os
import logging
from itertools import product  # Generate all combinations of variable parameters
import shap  # For feature importance analysis
import math
import copy  # For copying layer dictionaries

# ----------------------------
# 1. Setup and Configuration
# ----------------------------

# Suppress TensorFlow warnings for cleaner output
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
PLOTS_DIR = 'plots_01_02'
MODELS_DIR = 'models_01_02'
DATA_PATH = 'training_data.csv'

# Create necessary directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Define simulation parameters
resolution = 150        # pixels/um
dpml = 1                # PML thickness in um
pml_layers = [mp.PML(thickness=dpml)]

# Update: 12 wavelengths from 0.3 to 1.5 um and 5 plasmonic layer thicknesses from 0.01 to 0.05 um.
wavelengths_new = np.linspace(0.3, 1.5, 12)     # 12 wavelengths (um)
thicknesses_new = np.linspace(0.01, 0.05, 5)      # 5 thicknesses (um)

# Define available materials using Meep's built-in materials
materials_library = {
    'SiO2': SiO2,
    'Si3N4': Si3N4,
    'ITO': ITO,
    'Au': Au,
    'Ag': Ag,
    # Add more materials if needed
}

# Define fixed sandwich structure layers.
# In our structure, index 1 (the plasmonic layer) will be varied in thickness and material.
layers_fixed = [
    {'material': 'SiO2', 'thickness': 0.5},  # Substrate layer
    {'material': 'Au',   'thickness': 0.05},  # Plasmonic layer (to be varied)
    {'material': 'ITO',  'thickness': 0.2},   # Top layer
]

# Define range for variable layer parameters (only the plasmonic layer here, at index 1)
variable_layers = [
    {
        'index': 1,  # Plasmonic layer index
        'thickness_range': thicknesses_new,
        'materials': ['Au', 'Ag']
    }
]

# Generate all combinations (wavelength, plasmonic layer thickness).
# For each combination, we run two simulations (one with Au and one with Ag).
combinations = list(product(wavelengths_new, thicknesses_new))
num_simulations = len(combinations) * 2  # 2 materials per combination
print(f"Total number of simulations to run: {num_simulations}")

# Set simulation numbering to start from 1
starting_sim_id = 1
ending_sim_id = starting_sim_id + num_simulations - 1

# Global list to hold simulation records (to be flushed every 10 simulations)
data_records = []

# ----------------------------
# 2. Helper Function for Summary Plot
# ----------------------------

def update_summary_plot():
    """
    Reads the CSV file (if exists) and creates/updates a summary plot:
    Wavelength vs. Absorbed Power for Au and Ag, with separate curves for different plasmonic thicknesses.
    """
    if not os.path.isfile(DATA_PATH):
        print("No CSV file available for summary plot update.")
        return
    
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure that required columns exist
        required_cols = ['wavelength', 'layer_1_material', 'layer_1_thickness', 'absorbed_power']
        if not all(col in df.columns for col in required_cols):
            print("CSV does not have the required columns for summary plot.")
            return
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        for material, ax_idx in zip(['Au', 'Ag'], [0, 1]):
            subset = df[df['layer_1_material'] == material]
            if subset.empty:
                continue
            # Group by plasmonic layer thickness and plot wavelength vs absorbed power.
            for thickness, group in subset.groupby('layer_1_thickness'):
                group_sorted = group.sort_values('wavelength')
                ax[ax_idx].plot(group_sorted['wavelength'], group_sorted['absorbed_power'],
                                marker='o', linestyle='-', label=f"Thickness {thickness:.3f} μm")
            ax[ax_idx].set_title(f"{material} - Wavelength vs Absorbed Power", fontsize=10)
            ax[ax_idx].set_xlabel("Wavelength (μm)")
            ax[ax_idx].set_ylabel("Absorbed Power")
            ax[ax_idx].legend(fontsize=8)
            ax[ax_idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'wavelength_vs_absorbed_power.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Updated summary plot: wavelength_vs_absorbed_power.png")
    except Exception as e:
        print(f"Failed to update summary plot: {e}")

# ----------------------------
# 3. Simulation Function
# ----------------------------

def run_simulation(layers, wavelength, sim_id):
    """
    Runs a Meep simulation for a given multi-layer sandwich structure at a specific wavelength.
    
    Parameters:
    - layers (list of dict): List containing layer information with 'material' and 'thickness'.
    - wavelength (float): Wavelength in μm for the simulation.
    - sim_id (int): Simulation identifier.
    """
    # Calculate total thickness and set simulation cell dimensions
    total_thickness = sum(layer['thickness'] for layer in layers)
    y_max = total_thickness + 2 * dpml  # cell height (um)
    x_size = 4.0  # cell width (um)
    cell_size = mp.Vector3(x_size, y_max, 0)  # 2D simulation

    # Define geometry: create blocks for each layer
    geometry = []
    current_y = - (y_max / 2) + dpml
    for layer in layers:
        material = materials_library.get(layer['material'], mp.Medium(epsilon=1.0))
        thickness = layer['thickness']
        center = mp.Vector3(0, current_y + thickness / 2)
        block = mp.Block(
            material=material,
            center=center,
            size=mp.Vector3(mp.inf, thickness)
        )
        geometry.append(block)
        current_y += thickness

    # Define a source
    sources = [mp.Source(
        mp.GaussianSource(1 / wavelength, fwidth=0.1 * (1 / wavelength), is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(-x_size / 2 + dpml),
        size=mp.Vector3(0, y_max - 2 * dpml),
        amplitude=1.0
    )]

    # Initialize simulation
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        geometry=geometry
    )

    # Add DFT fields for absorption calculation
    dft_fields = sim.add_dft_fields(
        [mp.Dz, mp.Ez],
        1 / wavelength,
        0,
        1,
        center=mp.Vector3(),
        size=mp.Vector3(x_size, y_max),
        yee_grid=True
    )

    # Define flux regions for power calculation
    flux_box = sim.add_flux(
        1 / wavelength, 0, 1,
        mp.FluxRegion(center=mp.Vector3(x=-x_size / 2 + dpml, y=0), size=mp.Vector3(0, y_max - 2 * dpml), weight=+1),
        mp.FluxRegion(center=mp.Vector3(x=+x_size / 2 - dpml, y=0), size=mp.Vector3(0, y_max - 2 * dpml), weight=-1)
    )

    # Run the simulation (runtime increased for convergence)
    sim.run(until_after_sources=200)

    # Retrieve DFT field data and calculate absorbed power density
    Dz = sim.get_dft_array(dft_fields, mp.Dz, 0)
    Ez = sim.get_dft_array(dft_fields, mp.Ez, 0)
    absorbed_power_density = 2 * np.pi * (1 / wavelength) * np.imag(np.conj(Ez) * Dz)

    # Calculate absorbed power (integrate density over the simulation domain)
    dxy = (x_size / resolution) * (y_max / resolution)  # Area per pixel
    absorbed_power = np.sum(absorbed_power_density) * dxy
    absorbed_flux = mp.get_fluxes(flux_box)[0]
    err = abs(absorbed_power - absorbed_flux) / absorbed_flux

    # Print simulation summary
    material_names = [layer['material'] for layer in layers]
    print(f"Simulation {sim_id}: Wavelength={wavelength:.2f} μm, Layers={material_names}, "
          f"Absorbed Power={absorbed_power:.4f}, Absorbed Flux={absorbed_flux:.4f}, Error={err:.2%}")

    # Save cell structure plot
    plt.figure(figsize=(6, 6))
    sim.plot2D()
    plt.title(f"Simulation {sim_id}: Cell Structure", fontsize=10)
    plt.savefig(os.path.join(PLOTS_DIR, f'cell_structure_{sim_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save absorbed power density map
    y_coords, x_coords = np.meshgrid(
        np.linspace(-x_size / 2, x_size / 2, Dz.shape[1]),
        np.linspace(-y_max / 2, y_max / 2, Dz.shape[0]),
        indexing='ij'
    )
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        x_coords, y_coords, absorbed_power_density.T,
        cmap='inferno_r',
        shading='gouraud',
        vmin=0,
        vmax=np.amax(absorbed_power_density)
    )
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.gca().set_aspect('equal')
    # Add plasmonic layer thickness and material to the title and reduce font size.
    plasmonic_thickness = layers[1]['thickness']
    plasmonic_material = layers[1]['material']
    plt.title(f"Sim {sim_id}: Absorbed PD\nλ={wavelength:.2f} μm, {plasmonic_material}, t={plasmonic_thickness:.3f} μm", fontsize=10)
    plt.colorbar(label='Absorbed Power Density')
    plt.savefig(os.path.join(PLOTS_DIR, f'power_density_map_{sim_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Flatten the absorbed power density for ML target
    absorbed_power_flat = absorbed_power_density.flatten()

    # Prepare a record with features and target
    record = {
        'wavelength': wavelength,
        'absorbed_power': absorbed_power,  # Total absorbed power (for summary plot and regression)
        'absorbed_flux': absorbed_flux
    }
    # Save each layer's information (our plasmonic layer is at index 1)
    for idx, layer in enumerate(layers):
        record[f'layer_{idx}_material'] = layer['material']
        record[f'layer_{idx}_thickness'] = layer['thickness']
    # Save the spatial map as flattened pixel values
    for idx, val in enumerate(absorbed_power_flat):
        record[f'pixel_{idx}'] = val

    # Append the record to the global list
    data_records.append(record)

    # Reset simulation for next run
    sim.reset_meep()

# ----------------------------
# 4. Run Simulations with Periodic Data Flush
# ----------------------------

# (Optional) Check for an existing CSV file. If starting at simulation 1,
# warn the user if a file already exists.
if os.path.isfile(DATA_PATH):
    print(f"Note: '{DATA_PATH}' exists. New simulation data will be appended, even though simulation IDs start at 1.")

sim_id = starting_sim_id
# Loop over each (wavelength, thickness) combination and alternate between Au and Ag
for wavelength, thickness in combinations:
    for material in ['Au', 'Ag']:
        # Create a deep copy of the fixed layers and update the plasmonic layer (index 1)
        current_layers = copy.deepcopy(layers_fixed)
        current_layers[1]['thickness'] = thickness  # Set plasmonic layer thickness
        current_layers[1]['material'] = material    # Set plasmonic material

        # Run the simulation with current parameters
        run_simulation(current_layers, wavelength, sim_id)

        # After every 10 simulations, flush the data_records to CSV and update the summary plot
        if sim_id % 10 == 0:
            df_new = pd.DataFrame(data_records)
            if os.path.isfile(DATA_PATH):
                df_new.to_csv(DATA_PATH, mode='a', index=False, header=False)
            else:
                df_new.to_csv(DATA_PATH, index=False)
            data_records.clear()  # Clear records after flushing

            # Update summary plot of wavelength vs absorbed power for Au and Ag
            update_summary_plot()

        sim_id += 1

# Flush any remaining records (if total simulations is not an exact multiple of 10)
if data_records:
    df_new = pd.DataFrame(data_records)
    if os.path.isfile(DATA_PATH):
        df_new.to_csv(DATA_PATH, mode='a', index=False, header=False)
    else:
        df_new.to_csv(DATA_PATH, index=False)
    data_records.clear()
    update_summary_plot()

# ----------------------------
# 5. Machine Learning Pipeline
# ----------------------------

# Verify that the CSV file exists
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"The data file '{DATA_PATH}' does not exist.")

# Load the combined simulation data
df_ml = pd.read_csv(DATA_PATH)

print("\nCombined DataFrame Info:")
print(df_ml.info())
print("\nSample Data:")
print(df_ml.head())

# Check for missing values and fill if necessary
if df_ml.isnull().values.any():
    print("\nMissing values detected. Filling missing values with zeros.")
    df_ml.fillna(0, inplace=True)

# Ensure that the 'pixel_*' columns are numeric
pixel_columns = [col for col in df_ml.columns if col.startswith('pixel_')]
df_ml[pixel_columns] = df_ml[pixel_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert key features to numeric where needed
if not np.issubdtype(df_ml['wavelength'].dtype, np.number):
    df_ml['wavelength'] = pd.to_numeric(df_ml['wavelength'], errors='coerce').fillna(df_ml['wavelength'].mean())

for idx, layer in enumerate(layers_fixed):
    col_thickness = f'layer_{idx}_thickness'
    if col_thickness in df_ml.columns and not np.issubdtype(df_ml[col_thickness].dtype, np.number):
        df_ml[col_thickness] = pd.to_numeric(df_ml[col_thickness], errors='coerce').fillna(df_ml[col_thickness].mean())

# One-hot encode the material columns for each layer
layer_material_columns = [f'layer_{i}_material' for i in range(len(layers_fixed))]
for col in layer_material_columns:
    if col in df_ml.columns and not np.issubdtype(df_ml[col].dtype, np.number):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        material_encoded = encoder.fit_transform(df_ml[[col]])
        material_feature_names = encoder.get_feature_names_out([col])
        material_df = pd.DataFrame(material_encoded, columns=material_feature_names)
        df_ml = pd.concat([df_ml.drop(col, axis=1), material_df], axis=1)

# Define features and target for ML
# Features: all columns except pixel_* (which are the target spatial map)
feature_columns = [col for col in df_ml.columns if not col.startswith('pixel_')]
X = df_ml[feature_columns].values

# Target: all pixel_* columns (the flattened spatial map)
label_columns = [col for col in df_ml.columns if col.startswith('pixel_')]
y = df_ml[label_columns].values

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Feature Scaling
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Label Scaling
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(f"\nTraining Features shape: {X_train.shape}")
print(f"Testing Features shape: {X_test.shape}")
print(f"Training Labels shape: {y_train.shape}")
print(f"Testing Labels shape: {y_test.shape}")

# ----------------------------
# 5.1 Define the CNN Model
# ----------------------------

def build_cnn_model(input_dim, output_dim, dropout_rate=0.3):
    """
    Builds a Convolutional Neural Network model to predict absorbed power density maps.
    
    Parameters:
    - input_dim (int): Number of input features.
    - output_dim (int): Number of output labels (number of pixels).
    - dropout_rate (float): Dropout rate for regularization.
    
    Returns:
    - model (keras.Model): Compiled CNN model.
    """
    # Assume the output map is square; determine side length.
    map_size = int(math.sqrt(output_dim))
    if map_size * map_size != output_dim:
        raise ValueError("Output dimension is not a perfect square. Adjust map size accordingly.")
    
    inputs = keras.Input(shape=(input_dim,), name='input_layer')
    
    # Dense layers to project to a shape suitable for reshaping
    x = keras.layers.Dense(1024, activation='relu', name='dense_1024')(inputs)
    x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
    x = keras.layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    x = keras.layers.Dense(map_size * map_size, activation='linear', name='dense_output')(x)
    x = keras.layers.Reshape((map_size, map_size, 1), name='reshape')(x)
    
    # Convolutional layers for spatial refinement
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_1')(x)
    x = keras.layers.BatchNormalization(name='batch_norm_2')(x)
    x = keras.layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_2')(x)
    x = keras.layers.BatchNormalization(name='batch_norm_3')(x)
    x = keras.layers.Dropout(dropout_rate, name='dropout_3')(x)
    
    x = keras.layers.Conv2D(1, (3,3), activation='linear', padding='same', name='conv2d_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='CNN_AbsorbedPowerDensityModel')
    return model

# Build the CNN model
model = build_cnn_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
model.summary()

# ----------------------------
# 5.2 Compile and Train the Model
# ----------------------------

initial_learning_rate = 1e-3
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='mse',
              metrics=['mae'])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    lr_schedule
]

if X_train.shape[0] < 100:
    print("\nWarning: The training dataset is small. Consider running more simulations for better model performance.")

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# 5.3 Evaluate the Model
# ----------------------------

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest MAE: {test_mae}")
print(f"Test MSE: {test_loss}")

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(y_pred_scaled.shape[0], -1))
y_actual = scaler_y.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
print(f"Test RMSE: {rmse}")

# ----------------------------
# 5.4 Feature Importance with SHAP
# ----------------------------

shap_sample = X_train[:100]  # Adjust sample size as needed

try:
    explainer = shap.DeepExplainer(model, shap_sample)
    shap_values = explainer.shap_values(shap_sample)
    
    shap.summary_plot(shap_values[0], features=shap_sample, feature_names=feature_columns, show=False)
    plt.title("SHAP Summary Plot for First Pixel Prediction", fontsize=10)
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSHAP summary plot saved.")
except Exception as e:
    print(f"SHAP analysis failed: {e}")

# ----------------------------
# 5.5 Visualize Predictions vs Actual
# ----------------------------

num_plots = min(3, X_test.shape[0])
for i in range(num_plots):
    plt.figure(figsize=(12, 5))
    
    # Actual map
    plt.subplot(1, 2, 1)
    actual = y_actual[i].reshape(int(math.sqrt(y_actual.shape[1])), -1)
    plt.pcolormesh(
        np.linspace(-2, 2, actual.shape[1]),
        np.linspace(-2, 2, actual.shape[0]),
        actual,
        cmap='inferno_r',
        shading='gouraud'
    )
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.gca().set_aspect('equal')
    plt.title(f"Actual Absorbed Power Density\nSample {i+1}", fontsize=10)
    plt.colorbar(label='Absorbed Power Density')
    
    # Predicted map
    plt.subplot(1, 2, 2)
    predicted = y_pred[i].reshape(int(math.sqrt(y_pred.shape[1])), -1)
    plt.pcolormesh(
        np.linspace(-2, 2, predicted.shape[1]),
        np.linspace(-2, 2, predicted.shape[0]),
        predicted,
        cmap='inferno_r',
        shading='gouraud'
    )
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.gca().set_aspect('equal')
    plt.title(f"Predicted Absorbed Power Density\nSample {i+1}", fontsize=10)
    plt.colorbar(label='Absorbed Power Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'prediction_comparison_{i+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction comparison plot for Sample {i+1} saved.")

# ----------------------------
# 5.6 Save the Trained Model and Training History Plot
# ----------------------------

model.save(os.path.join(MODELS_DIR, 'absorbed_power_density_model_final.h5'))
print("\nTrained ML model saved as 'models/absorbed_power_density_model_final.h5'")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train MSE Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation MSE Loss', color='orange')
plt.plot(history.history['mae'], label='Train MAE', color='green')
plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss / MAE')
plt.title('Training and Validation Loss and MAE Over Epochs', fontsize=10)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'training_history_final.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Training history plot saved.")

# ----------------------------
# 6. Project Summary
# ----------------------------

print("""
Project Summary:
- Designed a multi-layer sandwich structure with configurable wavelengths and varying plasmonic layer thicknesses.
- Simulated electromagnetic absorption using Meep across 12 wavelengths (0.3 μm to 1.5 μm) and plasmonic layer thicknesses (0.01 μm to 0.05 μm).
- Alternated the plasmonic material between Gold (Au) and Silver (Ag) for each simulation.
- Generated and saved cell structure and absorbed power density maps in the 'plots' directory.
- Appended new simulation data to 'training_data.csv' after every 10 simulations to safeguard data.
- Included total absorbed power and flux along with a full spatial map (flattened) to clarify ML features and target.
- Loaded, preprocessed, and one-hot encoded the combined data for ML.
- Trained an enhanced CNN model capable of regenerating a spatial map and predicting absorbed power based on wavelength, plasmonic layer material (Au or Ag), and thickness.
- Utilized SHAP for feature importance analysis and visualized actual vs. predicted absorbed power density maps.
- Saved the trained model and training history for further analysis.
""")
