import logging
import random
import os
import gc
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across various libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Optional
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: str = "logs",
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        log_dir: Directory for log files
        log_filename: Log file name (default: {name}.log)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        filename = log_filename or f"{name}.log"
        file_path = Path(log_dir) / filename
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def reduce_mem_usage(df, verbose: bool = True) -> Any:
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: pandas DataFrame
        verbose: Print memory reduction info
    
    Returns:
        DataFrame with optimized dtypes
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df

def clear_memory() -> None:
    """
    Force garbage collection to free memory.
    """
    gc.collect()
    
def log_system_info(logger: logging.Logger) -> None:
    """
    Log system and environment information.
    
    Args:
        logger: Logger instance
    """
    import sys
    import platform
    
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info("="*60)
    
def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def format_time(seconds: float) -> str:
    """
    Format seconds to human readable time.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"
    
def export_dev_metadata(
    df,
    output_path: str = ".dev_metadata.json",
    project_name: str = "ml-base-project",
    notes: Optional[Dict[str, str]] = None,
    handling_strategy: Optional[Dict[str, str]] = None,
    extra_global_status: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export lightweight EDA metadata to JSON for CLI consumption.

    Args:
        df: pandas DataFrame (EDA dataset).
        output_path: Output JSON path.
        project_name: Project name label.
        notes: Optional per-feature notes {feature: note}.
        handling_strategy: Optional per-feature strategy {feature: strategy}.
        extra_global_stats: Optional extra global stats.
    """
    notes = notes or {}
    handling_strategy = handling_strategy or {}
    extra_global_status = extra_global_status or {}
    
    numeric_cols = df.select_dtypes(include='"number').columns.tolist()
    
    features = {}
    for col in df.columns:
        col_meta: Dict[str, Any] = {}
        col_meta["type"] = str(df[col].dtype)
        
        if col in numeric_cols:
            col_meta["mean"] = float(df[col].mean())
            col_meta["median"] = float(df[col].median())
            col_meta["std"] = float(df[col].std())
            col_meta["skewness"] = float(df[col].skew())
            col_meta["missing_pct"] = float(df[col].isna().mean())
            
            col_min = df[col].min()
            col_mix = df[col].max()
            if pd.notna(col_min) and pd.notna(col_mix):
                col_meta["range"] = [float(col_min), float(col_mix)]
        
        if col in notes:
            col_meta["notes"] = notes[col]
        if col in handling_strategy:
            col_meta["handling_strategy"] = handling_strategy[col]
        
        features[col] = col_meta
    
    global_stats = {
        "total_rows": int(len(df))
    }
    global_stats.update(extra_global_status)
    
    payload = {
        "project_name": project_name,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "features": features,
        "global_stats": global_stats
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)