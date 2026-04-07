#!/usr/bin/env python3
"""
KAN (Kolmogorov-Arnold Network) 10-Day Forecast Training Script
KAN uses learnable B-spline functions instead of fixed activations for enhanced non-linear modeling
Excellent for capturing complex climate patterns with better interpretability
"""

import sys
from datetime import datetime


def log_progress(message):
    """Print progress messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


# =============================================================================
# STEP 1: Import Libraries
# =============================================================================
log_progress("STEP 1/9: Importing required libraries...")
import pandas as pd
import polars as pl
import torch
from datetime import date
from neuralforecast import NeuralForecast
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import HuberLoss, MAE
log_progress("✓ Libraries imported successfully")


# =============================================================================
# STEP 2: Load Data
# =============================================================================
log_progress("STEP 2/9: Loading data from parquet file...")
output_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/data_together_at_daily/daily_time_series_with_features_updated.parquet'
pl_df = pl.read_parquet(output_path)
print(f"  Data shape: {pl_df.shape}")
print(f"  Data types: {pl_df.dtypes}")
print(f"  Preview:")
print(pl_df.head())
log_progress("✓ Data loaded successfully")


# =============================================================================
# STEP 3: Data Preprocessing
# =============================================================================
log_progress("STEP 3/9: Preprocessing data (renaming columns)...")
df = pl_df
df = df.rename({"spei_5d": "y", "station_no": "unique_id", "time": "ds"})

# Verify all required columns are present
required_cols = ['unique_id', 'ds', 'y', 'lat', 'lon', 'elevation', 
                 'temp_change', 'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                 'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                 'spei_change_15d', 'spei_change_30d', 'pcp']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print(f"  All required columns present: {required_cols}")
log_progress("✓ Data preprocessing completed")


# =============================================================================
# STEP 4: Train/Validation/Test Split
# =============================================================================
log_progress("STEP 4/9: Splitting data into train/validation/test sets...")
train_df = df.filter(
    (pl.col('ds') >= date(1979, 1, 5)) & (pl.col('ds') <= date(2010, 12, 31))
)
valid_df = df.filter(
    (pl.col('ds') >= date(2011, 1, 1)) & (pl.col('ds') <= date(2012, 12, 31))
)
test_df = df.filter(
    (pl.col('ds') >= date(2013, 1, 1)) & (pl.col('ds') <= date(2020, 12, 31))
)

val_size = valid_df.group_by('unique_id').agg(pl.len()).select(pl.col('len').min()).item()
test_size = test_df.group_by('unique_id').agg(pl.len()).select(pl.col('len').min()).item()
print(f"  Validation size: {val_size}, Test size: {test_size}")
log_progress("✓ Data split completed")


# =============================================================================
# STEP 5: Prepare Static Exogenous DataFrame
# =============================================================================
log_progress("STEP 5/9: Preparing static exogenous variables...")

# Create static DataFrame with unique station characteristics
static_df = df.select(['unique_id', 'lat', 'lon', 'elevation']).unique(subset=['unique_id'])
print(f"  Static DataFrame shape: {static_df.shape}")
print(f"  Number of unique stations: {static_df.shape[0]}")
print("  Static variables preview:")
print(static_df.head())
log_progress("✓ Static variables prepared")


# =============================================================================
# STEP 6: Model Configuration
# =============================================================================
log_progress("STEP 6/9: Configuring KAN hyperparameters...")

# Define exogenous variable lists
STAT_EXOG_LIST = ['lat', 'lon', 'elevation']
HIST_EXOG_LIST = ['temp_change', 'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                  'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                  'spei_change_15d', 'spei_change_30d', 'pcp']

# Forecasting Configuration
HORIZON = 10
INPUT_SIZE = 45  # Lookback window

# KAN Spline Configuration (Core KAN Parameters)
# These control the learnable B-spline basis functions
GRID_SIZE = 8  # Number of grid intervals for spline basis (5-32 typical range)
               # Higher = more flexibility but more parameters
               # Optimal for climate data: 8-16
SPLINE_ORDER = 3  # Order of B-spline (3=cubic, typical: 3-7)
                  # 3 is good balance between smoothness and flexibility

# KAN Spline Scaling Parameters (Fine-tuning spline behavior)
SCALE_NOISE = 0.1  # Noise level for initialization (0.1 is standard)
SCALE_BASE = 1.0   # Scale for base function (residual connection weight)
SCALE_SPLINE = 1.0  # Scale for spline function (main transformation weight)
ENABLE_STANDALONE_SCALE_SPLINE = True  # Allow independent spline scaling

# Grid Configuration for Spline Domain
GRID_EPS = 0.02  # Epsilon for grid boundary extension
GRID_RANGE = [-2, 2]  # Input range for splines (normalized)

# KAN Network Architecture
N_HIDDEN_LAYERS = 2  # Number of hidden KAN layers (1-3 typical for time series)
                     # More layers = more composition of functions
HIDDEN_SIZE = 256  # Hidden layer width (256-512 good for complex patterns)
                   # Each hidden unit learns a univariate spline function

# Training Configuration
LR = 1e-4  # KAN typically uses lower learning rate than MLPs
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
MAX_STEPS = 30_000
VAL_CHECK = 315
EARLY_STOP = 8  # Increased patience for KAN (slower convergence)

# Windowing Configuration
WINDOWS_BATCH_SIZE = 256
INFERENCE_WINDOWS_BATCH = 128

# Loss Configuration - Horizon weighting for long-term emphasis
horizon_weight = torch.tensor([1.0, 1.0, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0])

# Scaler Configuration
# KAN works well with 'robust' scaler for climate data with outliers
SCALER_TYPE = 'standard'

# Lightning Trainer Configuration
pl_trainer_cfg = dict(
    accelerator="gpu",
    devices=[0],
    precision='16-mixed',
    log_every_n_steps=100,
    gradient_clip_val=1.0,  # Important for KAN stability
    accumulate_grad_batches=2,  # Accumulate gradients for stability
    enable_checkpointing=True,
)

print(f"  Horizon: {HORIZON} days")
print(f"  Input size: {INPUT_SIZE} days")
print(f"  Grid size: {GRID_SIZE}, Spline order: {SPLINE_ORDER}")
print(f"  Hidden layers: {N_HIDDEN_LAYERS}, Hidden size: {HIDDEN_SIZE}")
print(f"  Scale parameters - Noise: {SCALE_NOISE}, Base: {SCALE_BASE}, Spline: {SCALE_SPLINE}")
print(f"  Grid range: {GRID_RANGE}, Grid epsilon: {GRID_EPS}")
print(f"  Static exogenous: {STAT_EXOG_LIST}")
print(f"  Historical exogenous: {len(HIST_EXOG_LIST)} variables")
print(f"  Learning rate: {LR}, Batch size: {BATCH_SIZE}")
print(f"  Scaler type: {SCALER_TYPE}")
log_progress("✓ Model configuration completed")


# =============================================================================
# STEP 7: Initialize KAN Model
# =============================================================================
log_progress("STEP 7/9: Initializing KAN model with learnable splines...")

model = KAN(
    # Forecasting task
    h=HORIZON,
    input_size=INPUT_SIZE,
    
    # KAN Spline Configuration (Core Innovation)
    grid_size=GRID_SIZE,  # Number of B-spline basis functions
    spline_order=SPLINE_ORDER,  # Order of B-spline (3=cubic)
    
    # Spline scaling parameters
    scale_noise=SCALE_NOISE,  # Initialization noise
    scale_base=SCALE_BASE,  # Base (residual) function weight
    scale_spline=SCALE_SPLINE,  # Spline function weight
    enable_standalone_scale_spline=ENABLE_STANDALONE_SCALE_SPLINE,
    
    # Grid parameters
    grid_eps=GRID_EPS,  # Grid boundary epsilon
    grid_range=GRID_RANGE,  # Spline input range
    
    # Network architecture
    n_hidden_layers=N_HIDDEN_LAYERS,  # Depth of KAN
    hidden_size=HIDDEN_SIZE,  # Width of hidden layers
    
    # Exogenous variables
    stat_exog_list=STAT_EXOG_LIST,
    hist_exog_list=HIST_EXOG_LIST,
    futr_exog_list=None,  # No future exogenous variables
    exclude_insample_y=False,  # Use historical target values
    
    # Loss function
    loss=HuberLoss(delta=0.5, horizon_weight=horizon_weight),
    valid_loss=MAE(),  # Use MAE for validation monitoring
    
    # Training parameters
    max_steps=MAX_STEPS,
    learning_rate=LR,
    num_lr_decays=-1,  # Automatic learning rate decay
    early_stop_patience_steps=EARLY_STOP,
    val_check_steps=VAL_CHECK,
    
    # Batch configuration
    batch_size=BATCH_SIZE,
    valid_batch_size=VALID_BATCH_SIZE,
    windows_batch_size=WINDOWS_BATCH_SIZE,
    inference_windows_batch_size=INFERENCE_WINDOWS_BATCH,
    
    # Data processing
    start_padding_enabled=False,
    step_size=1,
    scaler_type=SCALER_TYPE,
    random_seed=42,
    drop_last_loader=False,
    
    # Model alias
    alias='KAN_10day_drought_spline_based',
    
    # PyTorch Lightning trainer
    **pl_trainer_cfg,
)

log_progress("✓ KAN model initialized successfully")


# =============================================================================
# STEP 8: Training with Cross-Validation
# =============================================================================
log_progress("STEP 8/9: Starting KAN model training...")

nf = NeuralForecast(models=[model], freq='1d')

# Clear GPU cache
import gc
torch.cuda.empty_cache()
gc.collect()

log_progress("  Running cross-validation (training + evaluation)...")

# Select all required columns for cross-validation
cv_columns = ['unique_id', 'ds', 'y'] + HIST_EXOG_LIST

fcst_df = nf.cross_validation(
    df=df[cv_columns],  # Include all temporal and historical exogenous variables
    val_size=val_size,
    test_size=test_size,
    n_windows=None,  # Predict all possible rolling windows
    step_size=1,  # Predict once per day
    refit=False,  # Do not retrain at each window
    static_df=static_df,  # Pass static exogenous variables
)

log_progress("✓ Training and cross-validation completed")


# =============================================================================
# STEP 9: Save Model and Forecasts
# =============================================================================
log_progress("STEP 9/9: Saving model and forecast results...")

# Save the trained model
model_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/saved_models_multivariable/kan_10day_ahead_spline_based'
nf.save(
    path=model_save_path,
    model_index=None,
    overwrite=True,
    save_dataset=True
)
print(f"  Model saved to: {model_save_path}")

# Save forecast dataframe
fcst_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/fsct_df_multivariable/kan_fcst_df_10ahead_spline_based.parquet'
fcst_df.write_parquet(fcst_save_path)
print(f"  Forecasts saved to: {fcst_save_path}")

log_progress("✓ All results saved successfully")


# =============================================================================
# COMPLETION
# =============================================================================
log_progress("="*70)
log_progress("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
log_progress("="*70)
print(f"\nForecast DataFrame shape: {fcst_df.shape}")
print(f"\nForecast columns: {fcst_df.columns}")
print(f"\nFirst few predictions:")
print(fcst_df.head(20))

# Print memory usage
if torch.cuda.is_available():
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Additional statistics
print(f"\n{'='*70}")
print("PREDICTION STATISTICS")
print(f"{'='*70}")
print(f"Total predictions: {fcst_df.shape[0]}")
print(f"Unique stations: {fcst_df['unique_id'].n_unique()}")
print(f"\nPrediction range:")
print(f"  Min: {fcst_df['KAN_10day_drought_spline_based'].min():.4f}")
print(f"  Max: {fcst_df['KAN_10day_drought_spline_based'].max():.4f}")
print(f"  Mean: {fcst_df['KAN_10day_drought_spline_based'].mean():.4f}")
print(f"  Std: {fcst_df['KAN_10day_drought_spline_based'].std():.4f}")
print(f"\nActual values (y) range:")
print(f"  Min: {fcst_df['y'].min():.4f}")
print(f"  Max: {fcst_df['y'].max():.4f}")
print(f"  Mean: {fcst_df['y'].mean():.4f}")

# KAN-specific information
print(f"\n{'='*70}")
print("KAN ARCHITECTURE SUMMARY")
print(f"{'='*70}")
print(f"B-spline basis functions per layer: {GRID_SIZE + SPLINE_ORDER}")
print(f"Approximate learnable parameters: ~{HIDDEN_SIZE * (GRID_SIZE + SPLINE_ORDER) * N_HIDDEN_LAYERS:,}")
print(f"Spline composition depth: {N_HIDDEN_LAYERS} layers")
print(f"Non-linear function learning: Enabled via adaptive B-splines")
