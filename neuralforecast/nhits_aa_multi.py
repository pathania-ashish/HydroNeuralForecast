#!/usr/bin/env python3
"""
NHITS 10-Day Forecast Training Script - With Exogenous Variables (Memory Optimized)
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
import gc  # Move garbage collection import to top
from datetime import date
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import HuberLoss
log_progress("✓ Libraries imported successfully")


# =============================================================================
# STEP 2: Load Data with Memory Optimization
# =============================================================================
log_progress("STEP 2/9: Loading data from parquet file...")
output_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/data_together_at_daily/daily_time_series_with_features_updated.parquet'

# Verify all required columns are present
required_cols = ['station_no', 'time', 'spei_5d', 'lat', 'lon', 'elevation', 
                #  'spei_lag1', 'spei_lag2', 'spei_lag3', 'spei_lag4', 'spei_lag5',
                 'temp_change','spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                 'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                 'spei_change_15d', 'spei_change_30d', 'pcp']

pl_df = pl.read_parquet(output_path, columns=required_cols)
print(f"  Data shape: {pl_df.shape}")
log_progress("✓ Data loaded successfully")


# =============================================================================
# STEP 3: Data Preprocessing with Memory Optimization
# =============================================================================
log_progress("STEP 3/9: Preprocessing and optimizing data types...")
df = pl_df
df = df.rename({"spei_5d": "y", "station_no": "unique_id", "time": "ds"})

# Cast to float32 to save memory (float64 uses 2x memory)
numeric_cols = ['y', 'lat', 'lon', 'elevation', 
                'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                'spei_change_15d', 'spei_change_30d', 'pcp', 'temp_change']

for col in numeric_cols:
    if df[col].dtype == pl.Float64:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

# Delete original dataframe to free memory
del pl_df
gc.collect()

print(f"  Data shape after optimization: {df.shape}")
print(f"  Estimated memory: {df.estimated_size('mb'):.2f} MB")
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

# Delete split dataframes - we only need val_size and test_size
del train_df, valid_df, test_df
gc.collect()

log_progress("✓ Data split completed and intermediate dataframes released")


# =============================================================================
# STEP 5: Prepare Static Exogenous DataFrame
# =============================================================================
log_progress("STEP 5/9: Preparing static exogenous variables...")

# Create static DataFrame with unique station characteristics
static_df = df.select(['unique_id', 'lat', 'lon', 'elevation']).unique(subset=['unique_id'])

# Cast to float32
static_df = static_df.with_columns([
    pl.col('lat').cast(pl.Float32),
    pl.col('lon').cast(pl.Float32),
    pl.col('elevation').cast(pl.Float32)
])

print(f"  Static DataFrame shape: {static_df.shape}")
print(f"  Number of unique stations: {static_df.shape[0]}")
print(f"  Static DataFrame memory: {static_df.estimated_size('mb'):.2f} MB")
log_progress("✓ Static variables prepared")


# =============================================================================
# STEP 6: Model Configuration
# =============================================================================
log_progress("STEP 6/9: Configuring NHITS hyperparameters...")

# Define exogenous variable lists
STAT_EXOG_LIST = ['lat', 'lon', 'elevation']
HIST_EXOG_LIST = ['temp_change','spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                  'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                  'spei_change_15d', 'spei_change_30d', 'pcp']

# Forecasting Configuration
HORIZON = 10
INPUT_SIZE = 45

# NHITS Hierarchical Architecture
N_STACKS = 4
stack_types = ['identity'] * N_STACKS
n_blocks = [3, 3, 3, 2]

# MLP units per stack
mlp_units = [
    [384, 384],
    [384, 384],
    [384, 384],
    [384, 384]
]

# Pooling configuration
n_pool_kernel_size = [2, 4, 8, 16]
n_freq_downsample = [8, 4, 2, 1]
pooling_mode = 'MaxPool1d'
interpolation_mode = 'linear'

# Regularization
dropout_prob_theta = 0.15
activation = 'ReLU'

# Training Configuration
LR = 8e-4
BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
MAX_STEPS = 30_000
VAL_CHECK = 315
EARLY_STOP = 5
num_lr_decays = 3

# Windowing - REDUCED for memory efficiency
WINDOWS_BATCH_SIZE = 128  # Reduced from 256
INFERENCE_WINDOWS_BATCH = 64  # Reduced from 128

# Loss Configuration
horizon_weight = torch.tensor([1.0, 1.0, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0])

# Lightning Trainer Configuration
pl_trainer_cfg = dict(
    accelerator="gpu",
    devices=[0],
    precision='16-mixed',
    log_every_n_steps=100,
    gradient_clip_val=0.8,
    accumulate_grad_batches=1,
    enable_checkpointing=True,
)

print(f"  Horizon: {HORIZON} days")
print(f"  Input size: {INPUT_SIZE} days")
print(f"  Number of stacks: {N_STACKS}")
print(f"  Static exogenous: {STAT_EXOG_LIST}")
print(f"  Historical exogenous: {len(HIST_EXOG_LIST)} variables")
print(f"  Learning rate: {LR}, Batch size: {BATCH_SIZE}")
print(f"  Windows batch size: {WINDOWS_BATCH_SIZE} (reduced for memory)")
log_progress("✓ Model configuration completed")


# =============================================================================
# STEP 7: Memory Status Check Before Training
# =============================================================================
log_progress("STEP 7/9: Checking memory status before model initialization...")

import psutil
process = psutil.Process()
ram_usage = process.memory_info().rss / 1024**3
print(f"  Current RAM usage: {ram_usage:.2f} GB")
print(f"  Main dataframe memory: {df.estimated_size('mb'):.2f} MB")
print(f"  Static dataframe memory: {static_df.estimated_size('mb'):.2f} MB")

# Clear any remaining cached data
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log_progress("✓ Memory status checked and cleaned")


# =============================================================================
# STEP 8: Initialize Model
# =============================================================================
log_progress("STEP 8/9: Initializing NHITS model with exogenous variables...")

model = NHITS(
    # Forecasting task
    h=HORIZON,
    input_size=INPUT_SIZE,
    stat_exog_list=STAT_EXOG_LIST,
    hist_exog_list=HIST_EXOG_LIST,
    futr_exog_list=None,
    exclude_insample_y=False,
    
    # NHITS Hierarchical Architecture
    stack_types=stack_types,
    n_blocks=n_blocks,
    mlp_units=mlp_units,
    n_pool_kernel_size=n_pool_kernel_size,
    n_freq_downsample=n_freq_downsample,
    pooling_mode=pooling_mode,
    interpolation_mode=interpolation_mode,
    
    # Regularization
    dropout_prob_theta=dropout_prob_theta,
    activation=activation,
    
    # Loss function
    loss=HuberLoss(delta=0.5, horizon_weight=horizon_weight),
    valid_loss=None,
    
    # Training parameters
    max_steps=MAX_STEPS,
    learning_rate=LR,
    num_lr_decays=num_lr_decays,
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
    scaler_type='standard',
    random_seed=42,
    drop_last_loader=False,
    
    # Model alias
    alias='NHITS_10day_drought_with_exog',
    
    # PyTorch Lightning trainer
    **pl_trainer_cfg,
)

log_progress("✓ NHITS model initialized successfully")


# =============================================================================
# STEP 9: Training with Cross-Validation
# =============================================================================
log_progress("STEP 9/9: Starting NHITS model training...")

nf = NeuralForecast(models=[model], freq='1d')

# Final memory cleanup before training
gc.collect()
torch.cuda.empty_cache()

# Check memory one more time
ram_usage = process.memory_info().rss / 1024**3
print(f"  RAM usage before training: {ram_usage:.2f} GB")

log_progress("  Running cross-validation (training + evaluation)...")

# Select all required columns for cross-validation
cv_columns = ['unique_id', 'ds', 'y'] + HIST_EXOG_LIST

# Create training dataframe and immediately release original
df_training = df[cv_columns]
del df  # Delete full dataframe to free ~50% memory
gc.collect()

log_progress(f"  Training dataframe memory: {df_training.estimated_size('mb'):.2f} MB")

fcst_df = nf.cross_validation(
    df=df_training,
    val_size=val_size,
    test_size=test_size,
    n_windows=None,
    step_size=1,
    refit=False,
    static_df=static_df,
)

# Release training data after cross-validation
del df_training
gc.collect()

log_progress("✓ Training and cross-validation completed")


# =============================================================================
# STEP 10: Save Model and Forecasts
# =============================================================================
log_progress("STEP 10/10: Saving model and forecast results...")

# Save the trained model
model_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/saved_models_multivariable/nhits_10day_ahead_with_exog'
nf.save(
    path=model_save_path,
    model_index=None,
    overwrite=True,
    save_dataset=True
)
print(f"  Model saved to: {model_save_path}")

# Save forecast dataframe
fcst_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/fsct_df_multivariable/nhits_fcst_df_10ahead_with_exog.parquet'
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

# Final memory usage
if torch.cuda.is_available():
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

ram_usage = process.memory_info().rss / 1024**3
print(f"Final RAM usage: {ram_usage:.2f} GB")
