#!/usr/bin/env python3
"""
PatchTST 10-Day Forecast Training Script - Patch-based Transformer for Long-Horizon Forecasting
PatchTST uses subseries-level patches as input tokens to Transformers
Achieves state-of-the-art performance on long-term forecasting benchmarks (ICLR 2023)
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
from neuralforecast.models import PatchTST
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
log_progress("STEP 6/9: Configuring PatchTST hyperparameters...")

# Define exogenous variable lists
# STAT_EXOG_LIST = ['lat', 'lon', 'elevation']
STAT_EXOG_LIST = None
HIST_EXOG_LIST = ['temp_change', 'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                  'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                  'spei_change_15d', 'spei_change_30d', 'pcp']

# Forecasting Configuration
HORIZON = 10  # 10-day forecast
INPUT_SIZE = 45  # 45-day lookback window

# PatchTST Core Parameters (Key Innovation)
# ==========================================

# 1. Patching Configuration
PATCH_LEN = 5  # Length of each patch (subseries)
               # Divides 45-day history into patches
               # Optimal: 8-16 for daily data
               # Rule of thumb: patch_len ≈ input_size / (4 to 6)
               # 45 / 5 = 9 days per patch

STRIDE = 5  # Stride between consecutive patches
            # stride = patch_len → non-overlapping patches (more efficient)
            # stride < patch_len → overlapping patches (more information, slower)
            # Using non-overlapping for efficiency: 45 / 9 = 5 patches

# Number of patches = (input_size - patch_len) / stride + 1
# = (45 - 9) / 9 + 1 = 5 patches
# Each patch is treated as a "word" (token) for the Transformer

# 2. Reversible Instance Normalization (RevIN)
REVIN = True  # CRITICAL for non-stationary climate data
              # Normalizes each series, then denormalizes predictions
              # PatchTST paper shows huge improvement with RevIN
              # Always True for climate/weather data

AFFINE = True  # Learnable affine transformation in RevIN
SUBTRACT_LAST = False  # Alternative to RevIN (use either, not both)

# 3. Transformer Architecture
HIDDEN_SIZE = 256  # d_model: Transformer hidden dimension
                   # PatchTST paper uses 128-512
                   # Smaller than TFT because patches reduce sequence length
                   # 128 is efficient for 5 patches

N_HEADS = 16  # Number of attention heads
             # PatchTST paper recommends 8-16
             # Each head dimension: hidden_size / n_heads = 128 / 8 = 16

N_LAYERS = 4  # Number of Transformer encoder layers
              # PatchTST paper uses 3 layers for best results
              # More layers = better long-range dependencies

# 4. Feed-Forward Network
LINEAR_HIDDEN_SIZE = 512  # d_ff: FFN hidden dimension
                          # Typically 2-4x hidden_size
                          # 256 = 2 × 128

# 5. Regularization
DROPOUT = 0.2  # Main dropout rate
ATTN_DROPOUT = 0.1  # Attention dropout (separate)
FC_DROPOUT = 0.1  # Dropout in final projection head
HEAD_DROPOUT = 0.0  # Dropout before output head

# 6. Positional Encoding
PE = 'zeros'  # Positional encoding type: 'zeros', 'normal', 'uniform'
              # PatchTST paper uses 'zeros' (learned from scratch)
LEARN_PE = True  # Learn positional embeddings

# 7. Attention Mechanism
RES_ATTENTION = True  # Residual attention (PatchTST default)
                      # Adds residual connection to attention
PRE_NORM = False  # Pre-LayerNorm vs Post-LayerNorm
                  # False = Post-LN (PatchTST default)

# 8. Output Head
HEAD_TYPE = 'flatten'  # Type of prediction head
                       # 'flatten' = standard (recommended)
INDIVIDUAL = False  # Individual vs shared head across variables
                    # False = shared (more efficient)

# Training Configuration
LR = 3e-4  # PatchTST works well with standard learning rate
BATCH_SIZE = 64  # Match TFT
VALID_BATCH_SIZE = 64
MAX_STEPS = 30_000
VAL_CHECK = 315
EARLY_STOP = 5

# Learning Rate Scheduler
cosine_sched_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
cosine_sched_kwargs = dict(
    T_0=2500,
    T_mult=2,
    eta_min=1e-6,
)

# Windowing Configuration
WINDOWS_BATCH_SIZE = 256
INFERENCE_WINDOWS_BATCH = 128

# Loss Configuration - Horizon weighting
horizon_weight = torch.tensor([1.0, 1.0, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0])

# Scaler Configuration - MUST be 'standard'
SCALER_TYPE = 'identity'

# Lightning Trainer Configuration
pl_trainer_cfg = dict(
    accelerator="gpu",
    devices=[0],
    precision='32',
    log_every_n_steps=100,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    enable_checkpointing=True,
)

# Calculate and display patch information
num_patches = (INPUT_SIZE - PATCH_LEN) // STRIDE + 1
print(f"  Horizon: {HORIZON} days")
print(f"  Input size: {INPUT_SIZE} days")
print(f"  Patch configuration:")
print(f"    Patch length: {PATCH_LEN} days")
print(f"    Stride: {STRIDE} days")
print(f"    Number of patches: {num_patches}")
print(f"    Patch overlap: {'No' if STRIDE == PATCH_LEN else 'Yes'}")
print(f"  Transformer architecture:")
print(f"    Hidden size: {HIDDEN_SIZE}")
print(f"    Attention heads: {N_HEADS}")
print(f"    Encoder layers: {N_LAYERS}")
print(f"    FFN dimension: {LINEAR_HIDDEN_SIZE}")
print(f"  RevIN enabled: {REVIN}")
print(f"  Dropout rates - Main: {DROPOUT}, Attention: {ATTN_DROPOUT}")
print(f"  Static exogenous: {STAT_EXOG_LIST}")
print(f"  Historical exogenous: {len(HIST_EXOG_LIST)} variables")
print(f"  Learning rate: {LR}, Batch size: {BATCH_SIZE}")
print(f"  Scaler type: {SCALER_TYPE}")
log_progress("✓ Model configuration completed")


# =============================================================================
# STEP 7: Initialize PatchTST Model
# =============================================================================
log_progress("STEP 7/9: Initializing PatchTST model with patch-based architecture...")

model = PatchTST(
    # Forecasting task
    h=HORIZON,
    input_size=INPUT_SIZE,

    # Patching Configuration (Core PatchTST Innovation)
    patch_len=PATCH_LEN,  # Patch length
    stride=STRIDE,  # Stride between patches

    # Exogenous variables
    stat_exog_list=STAT_EXOG_LIST,
    hist_exog_list=HIST_EXOG_LIST,
    futr_exog_list=None,
    exclude_insample_y=False,

    # Reversible Instance Normalization
    revin=REVIN,  # Enable RevIN
    revin_affine=AFFINE,  # Learnable affine in RevIN
    revin_subtract_last=SUBTRACT_LAST,  # Alternative normalization

    # Transformer Architecture
    hidden_size=HIDDEN_SIZE,  # d_model
    n_heads=N_HEADS,  # Number of attention heads
    encoder_layers=N_LAYERS,  # Encoder layers

    # Feed-Forward Network
    linear_hidden_size=LINEAR_HIDDEN_SIZE,  # d_ff

    # Regularization
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    fc_dropout=FC_DROPOUT,
    head_dropout=HEAD_DROPOUT,

    # Positional Encoding
    learn_pos_embed=LEARN_PE,  # Learn positional embeddings

    # Attention Configuration
    res_attention=RES_ATTENTION,  # Residual attention
    batch_normalization=PRE_NORM,  # Pre vs Post LayerNorm

    # Activation
    activation='gelu',  # Activation function

    # Loss function
    loss=HuberLoss(delta=0.5, horizon_weight=horizon_weight),
    valid_loss=MAE(),
    
    # Training parameters
    max_steps=MAX_STEPS,
    learning_rate=LR,
    num_lr_decays=-1,
    early_stop_patience_steps=EARLY_STOP,
    val_check_steps=VAL_CHECK,
    
    # Learning rate scheduler
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
    scaler_type=SCALER_TYPE,
    random_seed=42,
    drop_last_loader=False,
    
    # Model alias
    alias='PatchTST_10day_drought_patches',
    
    # PyTorch Lightning trainer
    **pl_trainer_cfg,
)

log_progress("✓ PatchTST model initialized successfully")


# =============================================================================
# STEP 8: Training with Cross-Validation
# =============================================================================
log_progress("STEP 8/9: Starting PatchTST model training...")

nf = NeuralForecast(models=[model], freq='1d')

# Clear GPU cache
import gc
torch.cuda.empty_cache()
gc.collect()

log_progress("  Running cross-validation (training + evaluation)...")

# Select all required columns for cross-validation
cv_columns = ['unique_id', 'ds', 'y'] + HIST_EXOG_LIST

fcst_df = nf.cross_validation(
    df=df[cv_columns],
    val_size=val_size,
    test_size=test_size,
    n_windows=None,
    step_size=1,
    refit=False,
    static_df=None,
)

log_progress("✓ Training and cross-validation completed")


# =============================================================================
# STEP 9: Save Model and Forecasts
# =============================================================================
log_progress("STEP 9/9: Saving model and forecast results...")

# Save the trained model
model_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/saved_models_multivariable/patchtst_10day_ahead_patches'
nf.save(
    path=model_save_path,
    model_index=None,
    overwrite=True,
    save_dataset=True
)
print(f"  Model saved to: {model_save_path}")

# Save forecast dataframe
fcst_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/fsct_df_multivariable/patchtst_fcst_df_10ahead_patches.parquet'
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
    print(f"\nGPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Additional statistics
print(f"\n{'='*70}")
print("PREDICTION STATISTICS")
print(f"{'='*70}")
print(f"Total predictions: {fcst_df.shape[0]}")
print(f"Unique stations: {fcst_df['unique_id'].n_unique()}")
print(f"\nPrediction range:")
print(f"  Min: {fcst_df['PatchTST_10day_drought_patches'].min():.4f}")
print(f"  Max: {fcst_df['PatchTST_10day_drought_patches'].max():.4f}")
print(f"  Mean: {fcst_df['PatchTST_10day_drought_patches'].mean():.4f}")
print(f"  Std: {fcst_df['PatchTST_10day_drought_patches'].std():.4f}")
print(f"\nActual values (y) range:")
print(f"  Min: {fcst_df['y'].min():.4f}")
print(f"  Max: {fcst_df['y'].max():.4f}")
print(f"  Mean: {fcst_df['y'].mean():.4f}")

# PatchTST-specific information
print(f"\n{'='*70}")
print("PATCHTST ARCHITECTURE SUMMARY")
print(f"{'='*70}")
print(f"Patching mechanism: {INPUT_SIZE} days → {num_patches} patches of {PATCH_LEN} days")
print(f"Sequence reduction: {INPUT_SIZE} timesteps → {num_patches} tokens")
print(f"  Compression ratio: {INPUT_SIZE/num_patches:.1f}x")
print(f"  Attention complexity: O({num_patches}²) vs O({INPUT_SIZE}²) vanilla")
print(f"Channel-independence: Each station processed separately (shared weights)")
print(f"RevIN normalization: {'Enabled' if REVIN else 'Disabled'}")
print(f"Transformer: {N_LAYERS} layers × {N_HEADS} heads × {HIDDEN_SIZE//N_HEADS}-dim")
print(f"Total tokens processed: {num_patches} patches/series × {N_LAYERS} layers")
print(f"\nKey advantages:")
print(f"  ✓ {INPUT_SIZE/num_patches:.1f}x fewer tokens than vanilla Transformer")
print(f"  ✓ Captures local semantic information within patches")
print(f"  ✓ Superior long-term forecasting (ICLR 2023 benchmark winner)")
print(f"  ✓ Efficient channel-independence for large multi-series datasets")
