import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class BaseCAE(nn.Module):
    """
    Base class for Convolutional Autoencoder (CAE).
    """
    def __init__(self):
        super(BaseCAE, self).__init__()

    def encode(self, x):
        """
        Encodes the input using the encoder part of the CAE.
        """
        return self.encoder(x)

    def forward(self, x):
        """
        Forward pass of the CAE. Encodes and then decodes the input.
        """
        x = self.encoder(x)  # Encode the input
        x = self.decoder(x)  # Decode the encoded representation
        return x  # Return the reconstructed output


class simpleCAE(BaseCAE):
    """
    Simple Convolutional Autoencoder (CAE) with a basic encoder and decoder.
    """
    def __init__(self, input_channels=3):
        super(simpleCAE, self).__init__()

        # Simplified Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Simplified Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )


class simpleBatchCAE(BaseCAE):
    """
    Simple Convolutional Autoencoder (CAE) with batch normalization in both the encoder and decoder.
    """
    def __init__(self, input_channels=3):
        super(simpleBatchCAE, self).__init__()

        # Encoder with Batch Normalization
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),  # Batch normalization layer
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),  # Batch normalization layer
            nn.ReLU(),
        )

        # Decoder with Batch Normalization
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),  # Batch normalization layer
            nn.ReLU(),
            nn.ConvTranspose2d(12, input_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_channels),  # Batch normalization layer
            nn.Sigmoid(),
        )

class simpleBatchDropoutCAE(BaseCAE):
    """
    Simple Convolutional Autoencoder (CAE) with batch normalization and dropout in both the encoder and decoder.
    """
    def __init__(self, input_channels=3, dropout_rate=0.5):
        super(simpleBatchDropoutCAE, self).__init__()

        # Encoder with Batch Normalization and Dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),  # Batch normalization layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),  # Batch normalization layer
            nn.ReLU(),
        )

        # Decoder with Batch Normalization and Dropout
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),  # Batch normalization layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.ConvTranspose2d(12, input_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_channels),  # Batch normalization layer
            nn.Sigmoid(),
        )

class CAEWithPooling(BaseCAE):
    """
    Convolutional Autoencoder (CAE) with max pooling and corresponding upsampling layers.
    """
    def __init__(self, input_channels=3):
        super(CAEWithPooling, self).__init__()

        # Encoder with Max Pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=4, stride=2, padding=1),  # Output size: [12, x/2, y/2]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [12, x/4, y/4]
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),  # Output size: [24, x/8, y/8]
            nn.ReLU(),
            # No additional max pooling here to maintain symmetry with the decoder
        )

        # Decoder with Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # Upsample to [12, x/4, y/4]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to [12, x/2, y/2]
            nn.ConvTranspose2d(12, input_channels, kernel_size=4, stride=2, padding=1),  # Upsample to [3, x, y]
            nn.Sigmoid(),
        )

class DeepCAEWithPooling(BaseCAE):
    """
    Extended Convolutional Autoencoder (CAE) with additional max pooling and corresponding upsampling layers.
    """
    def __init__(self, input_channels=3):
        super(DeepCAEWithPooling, self).__init__()

        # Extended Encoder with More Max Pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output size: [12, x/2, y/2]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [12, x/4, y/4]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output size: [24, x/8, y/8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [24, x/16, y/16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output size: [48, x/32, y/32]
            nn.ReLU(),
        )

        # Extended Decoder with More Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to [48, x/16, y/16]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to [48, x/8, y/8]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample to [24, x/4, y/4]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to [24, x/2, y/2]
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Upsample to [input_channels, x, y]
            nn.Sigmoid(),
        )


class DeepCAE(BaseCAE):
    """
    Deep Convolutional Autoencoder (CAE) with multiple layers for more complex feature extraction.
    """
    def __init__(self, input_channels=3):
        super(DeepCAE, self).__init__()

        # Deep Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Deep Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

class DeepBatchCAE(BaseCAE):
    """
    Deep Convolutional Autoencoder (CAE) with multiple layers and batch normalization for more complex feature extraction.
    """
    def __init__(self, input_channels=3):
        super(DeepBatchCAE, self).__init__()

        # Deep Encoder with Batch Normalization
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Deep Decoder with Batch Normalization
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.Sigmoid(),
        )