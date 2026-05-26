"""
Time Series Decomposition Module
Classical and STL decomposition methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from config import OUTPUT_FIGURES_PATH, FIGURE_SIZE_LARGE, FIGURE_DPI
import os
import warnings
warnings.filterwarnings('ignore')

def classical_decompose(df, variable, period=365, model='additive', output_path=OUTPUT_FIGURES_PATH):
    """
    Perform classical seasonal decomposition.
    
    Args:
        df (pd.DataFrame): Input dataframe
        variable (str): Column name to decompose
        period (int): Seasonal period (365 for daily data with yearly seasonality)
        model (str): 'additive' or 'multiplicative'
        output_path (str): Path to save figure
        
    Returns:
        statsmodels.tsa.seasonal.DecompositionResult: Decomposition result
    """
    print(f"\n[DECOMPOSITION] Classical Decomposition for {variable} (period={period}, model={model})...")
    
    result = seasonal_decompose(df[variable], model=model, period=period, extrapolate='fill_mean')
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE_LARGE, sharex=True)
    
    axes[0].plot(df.index, result.observed, linewidth=1, color='steelblue')
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{variable} - Classical Decomposition (Additive)', fontsize=14, fontweight='bold')
    
    axes[1].plot(df.index, result.trend, linewidth=1.5, color='orange')
    axes[1].set_ylabel('Trend', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df.index, result.seasonal, linewidth=1, color='green')
    axes[2].set_ylabel('Seasonal', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(df.index, result.resid, linewidth=0.8, color='red')
    axes[3].set_ylabel('Residual', fontweight='bold')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_path, f'classical_decomposition_{variable}.png')
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[DECOMPOSITION] Saved: {filename}")
    plt.close()
    
    return result

def stl_decompose(df, variable, seasonal=365, trend=None, output_path=OUTPUT_FIGURES_PATH):
    """
    Perform STL (Seasonal and Trend decomposition using Loess) decomposition.
    STL is more robust than classical decomposition.
    
    Args:
        df (pd.DataFrame): Input dataframe
        variable (str): Column name to decompose
        seasonal (int): Seasonal window length
        trend (int): Trend window length (auto-calculated if None)
        output_path (str): Path to save figure
        
    Returns:
        statsmodels.tsa.seasonal.STLResult: STL decomposition result
    """
    print(f"\n[DECOMPOSITION] STL Decomposition for {variable} (seasonal={seasonal})...")
    
    # STL decomposition
    stl = STL(df[variable], seasonal=seasonal, trend=trend)
    result = stl.fit()
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE_LARGE, sharex=True)
    
    axes[0].plot(df.index, result.observed, linewidth=1, color='steelblue')
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{variable} - STL Decomposition', fontsize=14, fontweight='bold')
    
    axes[1].plot(df.index, result.trend, linewidth=1.5, color='orange')
    axes[1].set_ylabel('Trend', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df.index, result.seasonal, linewidth=1, color='green')
    axes[2].set_ylabel('Seasonal', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(df.index, result.resid, linewidth=0.8, color='red')
    axes[3].set_ylabel('Residual', fontweight='bold')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_path, f'stl_decomposition_{variable}.png')
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[DECOMPOSITION] Saved: {filename}")
    plt.close()
    
    return result

def plot_seasonal_subseries(df, variable, period=365, output_path=OUTPUT_FIGURES_PATH):
    """
    Plot seasonal subseries (one subplot per season).
    
    Args:
        df (pd.DataFrame): Input dataframe
        variable (str): Column name
        period (int): Seasonal period
        output_path (str): Path to save figure
    """
    print(f"\n[DECOMPOSITION] Creating seasonal subseries plot for {variable}...")
    
    # Create seasonal subseries
    df_temp = df[variable].copy()
    df_temp['season'] = df_temp.index.dayofyear % (period // 12)  # 12 seasons in a year
    
    # Group by season and plot
    n_seasons = min(12, len(df_temp.groupby('season')))
    fig, axes = plt.subplots(n_seasons, 1, figsize=(14, 12), sharex=False)
    
    if n_seasons == 1:
        axes = [axes]
    
    for i, (season, group) in enumerate(list(df_temp.groupby('season'))[:n_seasons]):
        axes[i].plot(group.index, group.values, linewidth=1, marker='o', markersize=3)
        axes[i].set_ylabel(f'Season {season}', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    fig.suptitle(f'{variable} - Seasonal Subseries Plot', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = os.path.join(output_path, f'seasonal_subseries_{variable}.png')
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[DECOMPOSITION] Saved: {filename}")
    plt.close()

def compare_decompositions(df, variable, period=365, output_path=OUTPUT_FIGURES_PATH):
    """
    Create side-by-side comparison of Classical and STL decompositions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        variable (str): Column name
        period (int): Seasonal period
        output_path (str): Path to save figure
    """
    print(f"\n[DECOMPOSITION] Comparing Classical vs STL for {variable}...")
    
    # Classical decomposition
    classical_result = seasonal_decompose(df[variable], model='additive', period=period, extrapolate='fill_mean')
    
    # STL decomposition
    stl_result = STL(df[variable], seasonal=period).fit()
    
    # Plot comparison
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    
    # Classical
    axes[0, 0].plot(df.index, classical_result.observed, linewidth=1, color='steelblue')
    axes[0, 0].set_ylabel('Observed', fontweight='bold')
    axes[0, 0].set_title('Classical Decomposition', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df.index, classical_result.trend, linewidth=1.5, color='orange')
    axes[1, 0].set_ylabel('Trend', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(df.index, classical_result.seasonal, linewidth=1, color='green')
    axes[2, 0].set_ylabel('Seasonal', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[3, 0].plot(df.index, classical_result.resid, linewidth=0.8, color='red')
    axes[3, 0].set_ylabel('Residual', fontweight='bold')
    axes[3, 0].set_xlabel('Date')
    axes[3, 0].grid(True, alpha=0.3)
    
    # STL
    axes[0, 1].plot(df.index, stl_result.observed, linewidth=1, color='steelblue')
    axes[0, 1].set_ylabel('Observed', fontweight='bold')
    axes[0, 1].set_title('STL Decomposition', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df.index, stl_result.trend, linewidth=1.5, color='orange')
    axes[1, 1].set_ylabel('Trend', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(df.index, stl_result.seasonal, linewidth=1, color='green')
    axes[2, 1].set_ylabel('Seasonal', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 1].plot(df.index, stl_result.resid, linewidth=0.8, color='red')
    axes[3, 1].set_ylabel('Residual', fontweight='bold')
    axes[3, 1].set_xlabel('Date')
    axes[3, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{variable} - Decomposition Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = os.path.join(output_path, f'decomposition_comparison_{variable}.png')
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[DECOMPOSITION] Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    from preprocessing import prepare_dataset
    
    df = load_and_prepare_data()
    df, _ = prepare_dataset(df)
    
    # Perform decompositions
    classical_result = classical_decompose(df, 'aqi', period=365)
    stl_result = stl_decompose(df, 'aqi', seasonal=365)
    compare_decompositions(df, 'aqi', period=365)
