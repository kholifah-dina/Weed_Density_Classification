import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    """
    Extract HSV, GLCM, and Hu Moments features from the segmented image.
    Following the layout from the research pipeline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # GLCM Properties
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    try:
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
    except Exception:
        # Fallback if image doesn't work correctly (e.g. blank/uniform patch after segmentation)
        zeros_4 = np.zeros(4)
        contrast = dissimilarity = homogeneity = energy = correlation = zeros_4

    # RGB and HSV Mean, Std Deviation
    rgb_mean = np.mean(image, axis=(0, 1))
    rgb_std = np.std(image, axis=(0, 1))
    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))

    # Hu Moments
    try:
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    except Exception:
        moments = np.zeros(7)

    # Stack all exactly as they were in the original dataframe
    # The dataframe had 39 features before selection
    # Indices map:
    # 0-3: contrast (0, 45, 90, 135)
    # 4-7: dissimilarity (0, 45, 90, 135)
    # 8-11: homogeneity (0, 45, 90, 135)
    # 12-15: energy (0, 45, 90, 135)
    # 16-19: correlation (0, 45, 90, 135)
    # 20-22: R_mean, G_mean, B_mean
    # 23-25: R_std, G_std, B_std
    # 26-28: H_mean, S_mean, V_mean
    # 29-31: H_std, S_std, V_std
    # 32-38: HuMoments 1 to 7
    all_features = np.hstack([
        contrast, dissimilarity, homogeneity, energy, correlation,
        rgb_mean, rgb_std, hsv_mean, hsv_std, moments
    ])

    return all_features

def select_features(all_features):
    """
    Select the Top 14 features from the Information Gain process validated in the research pipeline.
    """
    # Based on the selected indices from notebook 4_classification:
    # 'homogeneity_90deg' -> 10
    # 'homogeneity_45deg' -> 9
    # 'energy_45deg' -> 13
    # 'energy_135deg' -> 15
    # 'energy_90deg' -> 14
    # 'energy_0deg' -> 12
    # 'homogeneity_0deg' -> 8
    # 'homogeneity_135deg' -> 11
    # 'dissimilarity_90deg' -> 6
    # 'dissimilarity_135deg' -> 7
    # 'dissimilarity_45deg' -> 5
    # 'dissimilarity_0deg' -> 4
    # 'HuMoment_6' -> 37
    # 'G_mean' -> 21

    selected_indices = [10, 9, 13, 15, 14, 12, 8, 11, 6, 7, 5, 4, 37, 21]

    if len(all_features.shape) == 1:
        return all_features[selected_indices]
    else:
        return all_features[:, selected_indices]
