#!/usr/bin/env python3
"""
Informer 10-Day Forecast Training Script - ProbSparse Attention for Long-Horizon Forecasting
Informer uses ProbSparse self-attention achieving O(L log L) complexity
Specifically designed for Long Sequence Time-Series Forecasting (LSTF)
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
from neuralforecast.models import Informer
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
log_progress("STEP 6/9: Configuring Informer hyperparameters...")

# Define exogenous variable lists
STAT_EXOG_LIST = ['lat', 'lon', 'elevation']
HIST_EXOG_LIST = ['temp_change', 'spei_roll5_mean', 'spei_roll10_mean', 'spei_roll15_mean',
                  'wet_days_15d', 'wet_days_30d', 'spei_change_5d', 'spei_change_10d',
                  'spei_change_15d', 'spei_change_30d', 'pcp']

# Forecasting Configuration
HORIZON = 10  # 10-day forecast
INPUT_SIZE = 45  # 45-day lookback window

# Informer Architecture Parameters
# =================================

# 1. Model Dimensions
HIDDEN_SIZE = 256  # d_model: Dimension of model embeddings
                   # Must match TFT for fair comparison
                   # Typical range: 256-512

# 2. ProbSparse Self-Attention Configuration
N_HEAD = 4  # Number of attention heads
            # Informer paper recommends 8 heads for balance
            # Each head has dimension: hidden_size / n_head = 320/8 = 40

FACTOR = 3  # ProbSparse attention factor (unique to Informer)
            # Controls sparsity: selects top-u = c * ln(L) queries
            # c = factor, L = sequence length
            # Optimal range: 3-5 (3 for efficiency, 5 for accuracy)
            # Higher = more queries selected = less sparse

# 3. Encoder Architecture
ENCODER_LAYERS = 2  # Number of encoder layers with ProbSparse attention
                    # Informer paper uses 2-3 for long sequences
                    # More layers = better long-range dependency capture

# 4. Decoder Architecture  
DECODER_LAYERS = 1  # Number of decoder layers
                    # Typically 1-2 for Informer
                    # Decoder uses standard attention, not ProbSparse

# 5. Feed-Forward Network
D_FF = 512  # Dimension of feed-forward network
             # Standard: 4 * hidden_size = 4 * 320 = 1280
             # Using 1024 for efficiency (3.2x hidden_size)
             # Larger = more capacity but slower

# 6. Regularization
DROPOUT = 0.2  # Dropout rate
ATTENTION_DROPOUT = 0.1  # Attention dropout (separate from main dropout)

# 7. Activation Function
ACTIVATION = 'gelu'  # Activation for feed-forward networks
                     # 'gelu' is standard for Transformers
                     # Alternative: 'relu'

# 8. Positional Encoding
USE_NORM = True  # Use layer normalization
                 # Always True for stability (Informer default)

# Training Configuration
LR = 5e-4  # Informer typically uses lower LR than TFT
BATCH_SIZE = 64  # Match TFT
VALID_BATCH_SIZE = 64
MAX_STEPS = 30_000
VAL_CHECK = 315
EARLY_STOP = 5

# Learning Rate Scheduler - Cosine annealing
cosine_sched_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
cosine_sched_kwargs = dict(
    T_0=2500,
    T_mult=2,
    eta_min=1e-6,
)

# Windowing Configuration
WINDOWS_BATCH_SIZE = 256
INFERENCE_WINDOWS_BATCH = 128

# Loss Configuration - Horizon weighting for long-term emphasis
horizon_weight = torch.tensor([1.0, 1.0, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0])

# Scaler Configuration - MUST be 'standard' as per user requirement
SCALER_TYPE = 'standard'

# Lightning Trainer Configuration
pl_trainer_cfg = dict(
    accelerator="gpu",
    devices=[0],
    precision='32',
    log_every_n_steps=100,
    gradient_clip_val=1.0,  # Standard for Transformers
    accumulate_grad_batches=1,
    enable_checkpointing=True,
)

print(f"  Horizon: {HORIZON} days")
print(f"  Input size: {INPUT_SIZE} days")
print(f"  Hidden size (d_model): {HIDDEN_SIZE}")
print(f"  Attention heads: {N_HEAD}")
print(f"  ProbSparse factor: {FACTOR}")
print(f"  Encoder layers: {ENCODER_LAYERS}, Decoder layers: {DECODER_LAYERS}")
print(f"  Feed-forward dim: {D_FF}")
print(f"  Dropout: {DROPOUT}, Attention dropout: {ATTENTION_DROPOUT}")
print(f"  Activation: {ACTIVATION}")
print(f"  Static exogenous: {STAT_EXOG_LIST}")
print(f"  Historical exogenous: {len(HIST_EXOG_LIST)} variables")
print(f"  Learning rate: {LR}, Batch size: {BATCH_SIZE}")
print(f"  Scaler type: {SCALER_TYPE}")
log_progress("✓ Model configuration completed")


# =============================================================================
# STEP 7: Initialize Informer Model
# =============================================================================
log_progress("STEP 7/9: Initializing Informer model with ProbSparse attention...")

model = Informer(
    # Forecasting task
    h=HORIZON,
    input_size=INPUT_SIZE,
    
    # Exogenous variables
    stat_exog_list=STAT_EXOG_LIST,
    hist_exog_list=HIST_EXOG_LIST,
    futr_exog_list=None,  # No future exogenous variables
    exclude_insample_y=False,  # Use historical target values
    
    # Informer Architecture
    hidden_size=HIDDEN_SIZE,  # d_model dimension
    n_head=N_HEAD,  # Number of attention heads
    
    # ProbSparse Attention (Informer's Key Innovation)
    factor=FACTOR,  # Sparsity factor for ProbSparse attention
    
    # Encoder-Decoder Layers
    encoder_layers=ENCODER_LAYERS,  # Encoder depth
    decoder_layers=DECODER_LAYERS,  # Decoder depth
    
    # Feed-Forward Network
    conv_hidden_size=D_FF,  # FFN hidden dimension (called conv_hidden_size in Informer)

    # Regularization
    dropout=DROPOUT,  # Dropout (applies to both main and attention)

    # Activation
    activation=ACTIVATION,  # FFN activation function
    
    # Loss function
    loss=HuberLoss(delta=0.5, horizon_weight=horizon_weight),
    valid_loss=MAE(),  # Use MAE for validation monitoring
    
    # Training parameters
    max_steps=MAX_STEPS,
    learning_rate=LR,
    num_lr_decays=-1,  # Automatic learning rate decay
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
    scaler_type=SCALER_TYPE,  # MUST be 'standard'
    random_seed=42,
    drop_last_loader=False,
    
    # Model alias
    alias='Informer_10day_drought_probsparse',
    
    # PyTorch Lightning trainer
    **pl_trainer_cfg,
)

log_progress("✓ Informer model initialized successfully")


# =============================================================================
# STEP 8: Training with Cross-Validation
# =============================================================================
log_progress("STEP 8/9: Starting Informer model training...")

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
model_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/saved_models_multivariable/informer_10day_ahead_probsparse'
nf.save(
    path=model_save_path,
    model_index=None,
    overwrite=True,
    save_dataset=True
)
print(f"  Model saved to: {model_save_path}")

# Save forecast dataframe
fcst_save_path = '/home/ashish/my_data/Model_setup_FD/neuralforecast/neuralforecast/fsct_df_multivariable/informer_fcst_df_10ahead_probsparse.parquet'
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
print(f"  Min: {fcst_df['Informer_10day_drought_probsparse'].min():.4f}")
print(f"  Max: {fcst_df['Informer_10day_drought_probsparse'].max():.4f}")
print(f"  Mean: {fcst_df['Informer_10day_drought_probsparse'].mean():.4f}")
print(f"  Std: {fcst_df['Informer_10day_drought_probsparse'].std():.4f}")
print(f"\nActual values (y) range:")
print(f"  Min: {fcst_df['y'].min():.4f}")
print(f"  Max: {fcst_df['y'].max():.4f}")
print(f"  Mean: {fcst_df['y'].mean():.4f}")

# Informer-specific information
print(f"\n{'='*70}")
print("INFORMER ARCHITECTURE SUMMARY")
print(f"{'='*70}")
print(f"ProbSparse Attention: O(L log L) complexity vs O(L²) vanilla attention")
print(f"  Factor c={FACTOR}: Selects top-{FACTOR}*ln({INPUT_SIZE}) ≈ {int(FACTOR * 3.8)} queries")
print(f"  Attention heads: {N_HEAD} × {HIDDEN_SIZE//N_HEAD}-dim per head")
print(f"Encoder: {ENCODER_LAYERS} layers with self-attention distilling")
print(f"Decoder: {DECODER_LAYERS} layers with standard attention")
print(f"Feed-Forward: {D_FF}-dim (×{D_FF/HIDDEN_SIZE:.1f} expansion)")
print(f"Generative decoding: Predicts entire {HORIZON}-day sequence at once")
print(f"Memory savings: ~{100*(1 - FACTOR*3.8/INPUT_SIZE):.0f}% vs vanilla Transformer")
