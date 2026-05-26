"""
Stationarity Testing Module
Tests for stationarity using ADF, KPSS, and other methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, ndiffs
from config import ADF_SIGNIFICANCE_LEVEL, KPSS_SIGNIFICANCE_LEVEL, OUTPUT_TABLES_PATH, OUTPUT_FIGURES_PATH, FIGURE_SIZE_MEDIUM, FIGURE_DPI
import os

def adf_test(timeseries, name=''):
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Args:
        timeseries (pd.Series): Time series data
        name (str): Name for reporting
        
    Returns:
        dict: Test results
    """
    print(f"\n[STATIONARITY] ADF Test for {name}...")
    
    result = adfuller(timeseries.dropna(), autolag='AIC')
    
    test_result = {
        'test': 'ADF',
        'variable': name,
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'ic_best': result[5],
        'is_stationary': result[1] < ADF_SIGNIFICANCE_LEVEL
    }
    
    print(f"  - Test Statistic: {test_result['test_statistic']:.6f}")
    print(f"  - P-value: {test_result['p_value']:.6f}")
    print(f"  - Stationary: {'Yes' if test_result['is_stationary'] else 'No'}")
    print(f"  - Critical Values:")
    for key, value in test_result['critical_values'].items():
        print(f"      {key}: {value:.3f}")
    
    return test_result

def kpss_test(timeseries, name='', regression='c'):
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
    
    Args:
        timeseries (pd.Series): Time series data
        name (str): Name for reporting
        regression (str): 'c' for constant, 'ct' for constant and trend
        
    Returns:
        dict: Test results
    """
    print(f"\n[STATIONARITY] KPSS Test for {name}...")
    
    result = kpss(timeseries.dropna(), regression=regression, nlags='auto')
    
    test_result = {
        'test': 'KPSS',
        'variable': name,
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'critical_values': result[3],
        'is_stationary': result[1] > KPSS_SIGNIFICANCE_LEVEL
    }
    
    print(f"  - Test Statistic: {test_result['test_statistic']:.6f}")
    print(f"  - P-value: {test_result['p_value']:.6f}")
    print(f"  - Stationary: {'Yes' if test_result['is_stationary'] else 'No'}")
    print(f"  - Critical Values:")
    for key, value in test_result['critical_values'].items():
        print(f"      {key}: {value:.3f}")
    
    return test_result

def ndiff_test(timeseries, name='', alpha=0.05):
    """
    Determine number of differencing required for stationarity.
    
    Args:
        timeseries (pd.Series): Time series data
        name (str): Name for reporting
        alpha (float): Significance level
        
    Returns:
        dict: Test results
    """
    print(f"\n[STATIONARITY] Number of Differences Test for {name}...")
    
    # Using ADF-based approach
    n_diff_adf = ndiffs(timeseries.dropna(), alpha=alpha, test='adf')
    
    # Using KPSS-based approach
    n_diff_kpss = ndiffs(timeseries.dropna(), alpha=alpha, test='kpss')
    
    test_result = {
        'variable': name,
        'n_differences_adf': n_diff_adf,
        'n_differences_kpss': n_diff_kpss,
        'recommended': max(n_diff_adf, n_diff_kpss)
    }
    
    print(f"  - Differences (ADF): {n_diff_adf}")
    print(f"  - Differences (KPSS): {n_diff_kpss}")
    print(f"  - Recommended: {test_result['recommended']}")
    
    return test_result

def perform_stationarity_analysis(df, variables, output_path=OUTPUT_TABLES_PATH):
    """
    Perform comprehensive stationarity analysis on multiple variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        variables (list): List of column names to test
        output_path (str): Path to save results
        
    Returns:
        dict: Dictionary containing all test results
    """
    print("\n" + "="*80)
    print("STATIONARITY ANALYSIS")
    print("="*80)
    
    results = {
        'adf_results': [],
        'kpss_results': [],
        'ndiff_results': []
    }
    
    for var in variables:
        if var in df.columns:
            print(f"\n{'='*80}")
            print(f"Testing: {var.upper()}")
            print(f"{'='*80}")
            
            # ADF Test
            adf_result = adf_test(df[var], name=var)
            results['adf_results'].append(adf_result)
            
            # KPSS Test
            kpss_result = kpss_test(df[var], name=var)
            results['kpss_results'].append(kpss_result)
            
            # Number of differences
            ndiff_result = ndiff_test(df[var], name=var)
            results['ndiff_results'].append(ndiff_result)
    
    # Save results to CSV
    adf_df = pd.DataFrame(results['adf_results'])
    kpss_df = pd.DataFrame(results['kpss_results'])
    ndiff_df = pd.DataFrame(results['ndiff_results'])
    
    adf_file = os.path.join(output_path, 'adf_test_results.csv')
    kpss_file = os.path.join(output_path, 'kpss_test_results.csv')
    ndiff_file = os.path.join(output_path, 'ndiff_test_results.csv')
    
    adf_df.to_csv(adf_file, index=False)
    kpss_df.to_csv(kpss_file, index=False)
    ndiff_df.to_csv(ndiff_file, index=False)
    
    print(f"\n[STATIONARITY] Results saved:")
    print(f"  - {adf_file}")
    print(f"  - {kpss_file}")
    print(f"  - {ndiff_file}")
    
    print("\n" + "="*80)
    print("STATIONARITY ANALYSIS COMPLETED")
    print("="*80 + "\n")
    
    return results

def plot_stationarity_comparison(df, variable, output_path=OUTPUT_FIGURES_PATH):
    """
    Plot original series and differenced series for visual comparison.
    
    Args:
        df (pd.DataFrame): Input dataframe
        variable (str): Column name
        output_path (str): Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE_MEDIUM)
    
    # Original series
    axes[0].plot(df.index, df[variable], linewidth=1.5, color='steelblue')
    axes[0].set_title(f'{variable} - Original Series', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # First differencing
    diff1 = df[variable].diff().dropna()
    axes[1].plot(diff1.index, diff1, linewidth=1, color='coral')
    axes[1].set_title(f'{variable} - First Difference (d=1)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Difference')
    axes[1].grid(True, alpha=0.3)
    
    # Second differencing
    diff2 = df[variable].diff().diff().dropna()
    axes[2].plot(diff2.index, diff2, linewidth=1, color='lightgreen')
    axes[2].set_title(f'{variable} - Second Difference (d=2)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Difference')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_path, f'stationarity_comparison_{variable}.png')
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[STATIONARITY] Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    from preprocessing import prepare_dataset
    
    df = load_and_prepare_data()
    df, _ = prepare_dataset(df)
    
    results = perform_stationarity_analysis(df, ['aqi', 'pm2_5', 'pm10'])
