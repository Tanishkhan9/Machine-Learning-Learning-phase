import numpy as np
import pandas as pd
from scipy import stats

def advanced_statistics(data):
    arr = np.array(data)

    print("\n📊 Advanced Statistics Results 📊")
    print(f"Data: {arr}")
    
    # Basic statistics
    print(f"Mean: {np.mean(arr):.4f}")
    print(f"Median: {np.median(arr):.4f}")
    print(f"Mode: {stats.mode(arr, keepdims=True).mode[0]}")
    print(f"Variance: {np.var(arr, ddof=1):.4f}")
    print(f"Standard Deviation: {np.std(arr, ddof=1):.4f}")
    
    # Skewness & Kurtosis
    print(f"Skewness: {stats.skew(arr):.4f}")
    print(f"Kurtosis: {stats.kurtosis(arr):.4f}")
    
    # Quartiles & Percentiles
    print(f"25th Percentile (Q1): {np.percentile(arr, 25)}")
    print(f"50th Percentile (Median/Q2): {np.percentile(arr, 50)}")
    print(f"75th Percentile (Q3): {np.percentile(arr, 75)}")
    print(f"IQR (Q3-Q1): {np.percentile(arr, 75) - np.percentile(arr, 25)}")
    
    # Covariance & Correlation (if dataset is >1D)
    if arr.ndim == 2 and arr.shape[1] > 1:
        df = pd.DataFrame(arr)
        print("\nCovariance Matrix:\n", df.cov())
        print("\nCorrelation Matrix:\n", df.corr())

# Example run with 1D data
data = [10, 12, 23, 23, 16, 23, 21, 16, 34, 30, 23]
advanced_statistics(data)

# Example run with 2D data (for correlation/covariance)
# data2 = np.array([[10, 20, 30], [12, 18, 33], [23, 25, 40], [16, 22, 35]])
# advanced_statistics(data2)
