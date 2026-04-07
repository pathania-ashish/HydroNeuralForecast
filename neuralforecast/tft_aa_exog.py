"""
TFT (Temporal Fusion Transformer) 10-Day Forecast Training Script - With Exogenous Variables
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
log_progress("STEP 1/8: Importing required libraries...")
import pandas as pd
import polars as pl
import torch
from datetime import date
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberLoss
log_progress("✓ Libraries imported successfully")


# =============================================================================
# STEP 2: Load Data
# =============================================================================
log_progress("STEP 2/8: Loading data from parquet file...")
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
log_progress("STEP 3/8: Preprocessing data (renaming columns)...")
df = pl_df
df = df.rename({"spei_5d": "y", "station_no": "unique_id", "time": "ds"})

# Verify all required columns are present
required_cols = ['unique_id', 'ds', 'y', 'lat', 'lon', 'elevation', 
                #  'spei_lag1', 'spei_lag2', 'spei_lag3', 'spei_lag4', 'spei_lag5',
                 'temp_change','spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
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
log_progress("STEP 4/8: Splitting data into train/validation/test sets...")
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
# ===============================================================
log_progress("STEP 5/8: Preparing static exogenous variables...")

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
log_progress("STEP 6/8: Configuring TFT hyperparameters...")

# Define exogenous variable lists
STAT_EXOG_LIST = ['lat', 'lon', 'elevation']
# HIST_EXOG_LIST = ['spei_lag1', 'spei_lag2', 'spei_lag3', 'spei_lag4', 'spei_lag5',
#                   'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
#                   'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
#                   'spei_change_15d', 'spei_change_30d', 'pcp', 'tmax', 'tmin']
HIST_EXOG_LIST = ['temp_change','spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                  'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                  'spei_change_15d', 'spei_change_30d', 'pcp']
# Forecasting Configuration
HORIZON = 10
INPUT_SIZE = 45

# Architecture - Increased hidden size for more features
HIDDEN_SIZE = 320  # Increased from 256 to handle more features
N_HEAD = 8

# LSTM Encoder
RNN_TYPE = "lstm"
N_RNN_LAYERS = 2
ONE_RNN_INITIAL_STATE = False

# Gating and Dropout
GRN_ACTIVATION = "ELU"
DROPOUT = 0.2  # Slightly increased for regularization with more features
ATTN_DROPOUT = 0.1

# Training Configuration
LR = 8e-4  # Slightly reduced for stability with more features
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
MAX_STEPS = 30_000  # Increased for more complex model
VAL_CHECK = 315
EARLY_STOP = 5

# Learning Rate Scheduler
cosine_sched_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
cosine_sched_kwargs = dict(
    T_0=2500,
    T_mult=2,
    eta_min=1e-6,
)

# Windowing
WINDOWS_BATCH_SIZE = 256
INFERENCE_WINDOWS_BATCH = 128

# Loss Configuration - Moderate weighting
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
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  LSTM layers: {N_RNN_LAYERS}, Attention heads: {N_HEAD}")
print(f"  Static exogenous: {STAT_EXOG_LIST}")
print(f"  Historical exogenous: {len(HIST_EXOG_LIST)} variables")
print(f"  Learning rate: {LR}, Batch size: {BATCH_SIZE}")
print(f"  Max steps: {MAX_STEPS}")
log_progress("✓ Model configuration completed")


# =============================================================================
# STEP 7: Initialize Model
# =============================================================================
log_progress("STEP 7/8: Initializing TFT model with exogenous variables...")

model = TFT(
    # Forecasting task
    h=HORIZON,
    input_size=INPUT_SIZE,
    stat_exog_list=STAT_EXOG_LIST,
    hist_exog_list=HIST_EXOG_LIST,
    futr_exog_list=None,  # No future exogenous variables
    
    # Architecture
    hidden_size=HIDDEN_SIZE,
    n_head=N_HEAD,
    
    # LSTM Encoder
    rnn_type=RNN_TYPE,
    n_rnn_layers=N_RNN_LAYERS,
    one_rnn_initial_state=ONE_RNN_INITIAL_STATE,
    
    # Gating and regularization
    grn_activation=GRN_ACTIVATION,
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    
    # Loss function
    loss=HuberLoss(delta=0.5, horizon_weight=horizon_weight),
    valid_loss=None,
    
    # Training parameters
    max_steps=MAX_STEPS,
    learning_rate=LR,
    num_lr_decays=-1,
    early_stop_patience_steps=EARLY_STOP,
    val_check_steps=VAL_CHECK,
    
    # Optimizer and scheduler
    lr_scheduler=cosine_sched_cls,
    lr_scheduler_kwargs=cosine_sched_kwargs,
    
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
    alias='TFT_10day_drought_with_exog',
    
    # PyTorch Lightning trainer
    **pl_trainer_cfg,
)

log_progress("✓ TFT model initialized successfully")


# =============================================================================
# STEP 8: Training with Cross-Validation
# =============================================================================
log_progress("STEP 8/8: Starting TFT model training...")

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
model_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/saved_models_multivariable/tft_10day_ahead_with_exog'
nf.save(
    path=model_save_path,
    model_index=None,
    overwrite=True,
    save_dataset=True
)
print(f"  Model saved to: {model_save_path}")

# Save forecast dataframe
fcst_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/fsct_df_multivariable/tft_fcst_df_10ahead_with_exog.parquet'
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
