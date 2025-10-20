
"""
Data Augmentation Module for Time Series Anomaly Detection

This module provides data augmentation techniques for time series data, specifically designed
for anomaly detection tasks. It includes two main augmentation strategies:

1. NoiseTransformation: Adds Gaussian noise to time series data
2. SubAnomaly: Injects various types of synthetic anomalies into time series windows

The augmentations are designed to improve model robustness and generalization by creating
diverse training examples with different anomaly patterns.
"""

import random
import numpy as np
import torch

# Set device for tensor operations (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseTransformation(object):
    """
    Adds Gaussian noise to time series data for data augmentation.
    
    This transformation helps improve model robustness by adding controlled noise
    to the input data, making the model more resilient to small variations
    and measurement errors in real-world data.
    
    Args:
        sigma (float): Standard deviation of the Gaussian noise distribution.
                      Higher values result in more noise being added.
    """
    
    def __init__(self, sigma):
        """
        Initialize the noise transformation with specified standard deviation.
        
        Args:
            sigma (float): Standard deviation for Gaussian noise generation
        """
        self.sigma = sigma

    def __call__(self, X):
        """
        Apply Gaussian noise transformation to input tensor.
        
        The method adds random Gaussian noise with mean 0 and standard deviation
        equal to self.sigma to the input tensor. The operation handles GPU/CPU
        tensor movement automatically.
        
        Args:
            X (torch.Tensor): Input time series tensor
            
        Returns:
            torch.Tensor: Input tensor with added Gaussian noise
        """
        # Check if input tensor is on GPU and move to CPU for NumPy operations
        if X.device.type == 'cuda':  # Check if X is on GPU
            X = X.cpu()  # Move tensor to CPU
            
        # Generate Gaussian noise with specified standard deviation
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)  # NumPy operation
        
        # Add noise to original data and return as PyTorch tensor on original device
        return torch.tensor(X.numpy() + noise, dtype=torch.float32, device=device)  # Move back to GPU

class SubAnomaly(object):
    """
    Injects synthetic anomalies into time series data for anomaly detection training.
    
    This class creates various types of synthetic anomalies by manipulating subsequences
    of the input time series. It supports multiple anomaly types including seasonal,
    trend, global, contextual, and shapelet anomalies. This helps train models to
    recognize different anomaly patterns in real-world data.
    
    Args:
        portion_len (int): Length of the portion to be modified (currently unused parameter)
    """
    
    def __init__(self, portion_len):
        """
        Initialize the SubAnomaly transformation.
        
        Args:
            portion_len (int): Length parameter for anomaly injection (currently unused)
        """
        self.portion_len = portion_len

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int= None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None
                                 ):
        """
        Injects a synthetic anomaly into a time series window by manipulating a subsequence.
        
        This method creates various types of anomalies by:
        1. Selecting a random subsequence from the input window
        2. Applying compression, scaling, and trend transformations
        3. Replacing the original subsequence with the modified one
        
        Args:
            window (torch.Tensor): The time series window to modify (2D tensor)
            subsequence_length (int, optional): Length of subsequence to manipulate.
                                              If None, randomly chosen between 10%-90% of window length
            compression_factor (int, optional): Factor for compressing the subsequence.
                                              If None, randomly chosen between 2-5
            scale_factor (float, optional): Factor for scaling the subsequence values.
                                           If None, randomly chosen between 0.1-2.0 for each feature
            trend_factor (float, optional): Factor for adding trend to the subsequence.
                                          If None, randomly sampled from normal distribution
            shapelet_factor (bool): If True, adds shapelet-based anomaly pattern
            trend_end (bool): If True, extends anomaly to the end of the window
            start_index (int, optional): Starting index for the subsequence.
                                        If None, randomly chosen
            
        Returns:
            torch.Tensor: Modified window with injected anomaly
        """

        # Clone the input tensor to avoid modifying the original data
        window = window.clone() #.copy()

        # Set the subsequence_length if not provided
        # Choose random length between 10% and 90% of the window length
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)  # Minimum 10% of window length
            max_len = int(window.shape[0] * 0.9)  # Maximum 90% of window length
            subsequence_length = np.random.randint(min_len, max_len)

        # Set the compression_factor if not provided
        # Compression factor determines how much the subsequence is compressed
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)  # Random compression between 2-5x

        # Set the scale_factor if not provided
        # Scale factor determines the magnitude change of the subsequence
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])  # Random scale for each feature
            print('test')  # Debug print (can be removed)

        # Randomly select the start index for the subsequence
        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        
        # Calculate end index, ensuring it doesn't exceed window bounds
        end_index = min(start_index + subsequence_length, window.shape[0])

        # If trend_end is True, extend the anomaly to the end of the window
        if trend_end:
            end_index = window.shape[0]

        # Extract the subsequence from the window that will be modified
        anomalous_subsequence = window[start_index:end_index]

        # Apply compression transformation
        # This creates a compressed version by repeating and subsampling
        # anomalous_subsequence = np.tile(anomalous_subsequence, (compression_factor, 1))
        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)  # PyTorch equivalent of np.tile()
        anomalous_subsequence = anomalous_subsequence[::compression_factor]  # Subsample to compress

        # Apply scaling transformation to change the magnitude of values
        anomalous_subsequence = anomalous_subsequence * scale_factor

        # Apply trend transformation
        # Add a linear trend to the subsequence
        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)  # Random trend factor from normal distribution
        
        # Randomly choose trend direction (positive or negative)
        coef = 1
        if np.random.uniform() < 0.5: 
            coef = -1  # 50% chance of negative trend
        
        # Add the trend to the subsequence
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        # Apply shapelet transformation if enabled
        # Shapelet factor creates a pattern-based anomaly
        if shapelet_factor:
            # Create shapelet pattern by adding small random variations to the starting point
            # anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape(-1, 1)
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)  # GPU-compatible version

        # Replace the original subsequence with the modified anomalous subsequence
        window[start_index:end_index] = anomalous_subsequence

        # Return the modified window, removing any unnecessary dimensions
        return np.squeeze(window)

    def __call__(self, X):
        """
        Apply SubAnomaly transformation by randomly injecting one of five anomaly types.
        
        This method creates five different types of synthetic anomalies and randomly
        selects one to apply to the input time series:
        
        1. Seasonal Anomaly: Frequency-based anomalies with compression
        2. Trend Anomaly: Linear trend anomalies extending to window end
        3. Global Anomaly: Large magnitude changes in short subsequences
        4. Contextual Anomaly: Moderate magnitude changes in medium subsequences
        5. Shapelet Anomaly: Pattern-based anomalies using shapelet transformations
        
        Args:
            X (torch.Tensor): Input time series tensor (1D or 2D)
            
        Returns:
            torch.Tensor: Time series with randomly injected anomaly
        """
        # Clone input tensor to avoid modifying original data
        window = X.clone() #X.copy()
        
        # Create copies for each anomaly type
        anomaly_seasonal = window.clone()   # Seasonal/frequency-based anomalies
        anomaly_trend = window.clone()      # Trend-based anomalies
        anomaly_global = window.clone()     # Global magnitude anomalies
        anomaly_contextual = window.clone() # Contextual anomalies
        anomaly_shapelet = window.clone()   # Shapelet-based anomalies
        
        # Calculate random subsequence parameters
        min_len = int(window.shape[0] * 0.1)  # Minimum 10% of window length
        max_len = int(window.shape[0] * 0.9)  # Maximum 90% of window length
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(window) - subsequence_length)
        
        # Handle multivariate time series (2D tensor)
        if (window.ndim > 1):
            num_features = window.shape[1]  # Number of features/dimensions
            
            # Randomly select number of dimensions to modify (10%-50% of features)
            num_dims = np.random.randint(int(num_features/10), int(num_features/2))
            
            # Apply different anomaly types to randomly selected features
            for k in range(num_dims):
                i = np.random.randint(0, num_features)  # Random feature index
                temp_win = window[:, i].reshape((window.shape[0], 1))  # Extract single feature
                
                # 1. SEASONAL ANOMALY: Frequency-based compression without scaling or trend
                anomaly_seasonal[:, i] = self.inject_frequency_anomaly(temp_win,
                                                              scale_factor=1,      # No scaling
                                                              trend_factor=0,      # No trend
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                # 2. TREND ANOMALY: Linear trend extending to window end
                anomaly_trend[:, i] = self.inject_frequency_anomaly(temp_win,
                                                             compression_factor=1, # No compression
                                                             scale_factor=1,       # No scaling
                                                             trend_end=True,       # Extend to end
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                # 3. GLOBAL ANOMALY: Large magnitude change in very short subsequence
                anomaly_global[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=2,   # Very short
                                                            compression_factor=1,  # No compression
                                                            scale_factor=8,         # Large scaling
                                                            trend_factor=0,         # No trend
                                                           start_index = start_index)

                # 4. CONTEXTUAL ANOMALY: Moderate magnitude change in medium subsequence
                anomaly_contextual[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=4,   # Medium length
                                                            compression_factor=1,  # No compression
                                                            scale_factor=3,         # Moderate scaling
                                                            trend_factor=0,         # No trend
                                                           start_index = start_index)

                # 5. SHAPELET ANOMALY: Pattern-based anomaly using shapelet transformation
                anomaly_shapelet[:, i] = self.inject_frequency_anomaly(temp_win,
                                                          compression_factor=1,    # No compression
                                                          scale_factor=1,          # No scaling
                                                          trend_factor=0,          # No trend
                                                          shapelet_factor=True,    # Enable shapelet
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

        # Handle univariate time series (1D tensor)
        else:
            temp_win = window.reshape((len(window), 1))  # Reshape to 2D for processing
            
            # Apply same anomaly types but with different parameters for univariate data
            # 1. SEASONAL ANOMALY
            anomaly_seasonal = self.inject_frequency_anomaly(temp_win,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

            # 2. TREND ANOMALY
            anomaly_trend = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True,
                                                         subsequence_length=subsequence_length,
                                                         start_index = start_index)

            # 3. GLOBAL ANOMALY (slightly longer subsequence for univariate)
            anomaly_global = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,    # Longer than multivariate
                                                        compression_factor=1,
                                                        scale_factor=8,
                                                        trend_factor=0,
                                                        start_index = start_index)

            # 4. CONTEXTUAL ANOMALY (slightly longer subsequence for univariate)
            anomaly_contextual = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,    # Longer than multivariate
                                                        compression_factor=1,
                                                        scale_factor=3,
                                                        trend_factor=0,
                                                        start_index = start_index)

            # 5. SHAPELET ANOMALY
            anomaly_shapelet = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True,
                                                      subsequence_length=subsequence_length,
                                                      start_index = start_index)

        # Create list of all anomaly types
        anomalies = [anomaly_seasonal,    # Seasonal/frequency anomalies
                     anomaly_trend,       # Trend anomalies
                     anomaly_global,      # Global magnitude anomalies
                     anomaly_contextual,  # Contextual anomalies
                     anomaly_shapelet     # Shapelet pattern anomalies
                     ]

        # Randomly select one anomaly type to return
        anomalous_window = random.choice(anomalies)

        return anomalous_window



