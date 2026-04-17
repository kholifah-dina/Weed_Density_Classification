import cv2
import numpy as np


def extract_features(image):
    """
    Ekstraksi 19 fitur dari gambar tersegmentasi: RGB (6) + HSV (6) + Hu Moments (7).
    Tidak menggunakan GLCM. Semua 19 fitur langsung digunakan untuk training.

    Urutan kolom (sesuai Data_ekstraksi_Fitur_Gulma.csv — kolom non-GLCM):
      0  : R_mean      6  : H_mean      12 : HuMoment_1
      1  : G_mean      7  : S_mean      13 : HuMoment_2
      2  : B_mean      8  : V_mean      14 : HuMoment_3
      3  : R_std       9  : H_std       15 : HuMoment_4
      4  : G_std       10 : S_std       16 : HuMoment_5
      5  : B_std       11 : V_std       17 : HuMoment_6
                                        18 : HuMoment_7
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RGB Mean & Std (idx 0-5)
    rgb_mean = np.mean(image, axis=(0, 1))
    rgb_std  = np.std(image,  axis=(0, 1))

    # HSV Mean & Std (idx 6-11)
    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std  = np.std(hsv,  axis=(0, 1))

    # Hu Moments (idx 12-18)
    try:
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    except Exception:
        moments = np.zeros(7)

    # Stack 19 fitur
    return np.hstack([rgb_mean, rgb_std, hsv_mean, hsv_std, moments])


def extract_features_with_glcm(image):
    """
    Ekstraksi 39 fitur dari gambar tersegmentasi:
      - 20 fitur GLCM (5 properti × 4 sudut: 0°, 45°, 90°, 135°)
      - 6  fitur RGB  (mean & std)
      - 6  fitur HSV  (mean & std)
      - 7  fitur Hu Moments
    Total: 39 fitur

    Urutan kolom sesuai CSV 39 fitur (Data_ekstraksi_Fitur_Gulma.csv):
      0-3   : contrast_0deg .. contrast_135deg
      4-7   : dissimilarity_0deg .. dissimilarity_135deg
      8-11  : homogeneity_0deg .. homogeneity_135deg
      12-15 : energy_0deg .. energy_135deg
      16-19 : correlation_0deg .. correlation_135deg
      20-25 : R_mean, G_mean, B_mean, R_std, G_std, B_std
      26-31 : H_mean, S_mean, V_mean, H_std, S_std, V_std
      32-38 : HuMoment_1 .. HuMoment_7

    Digunakan saat inference dengan model yang dilatih dari CSV 39 fitur
    (pipeline menyimpan 'selector' untuk seleksi LAN).
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        raise ImportError(
            "scikit-image diperlukan untuk ekstraksi GLCM. "
            "Jalankan: pip install scikit-image"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── GLCM features (20) ──────────────────────────────────────────────
    # 4 sudut: 0°, 45°, 90°, 135°  →  [0, π/4, π/2, 3π/4]
    glcm = graycomatrix(
        gray, distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256, symmetric=True, normed=True,
    )
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        # graycoprops returns shape (n_distances, n_angles) → (1, 4)
        vals = graycoprops(glcm, prop)[0]   # shape (4,)
        glcm_features.extend(vals.tolist())  # 0°, 45°, 90°, 135°

    # ── RGB Mean & Std (6) ──────────────────────────────────────────────
    rgb_mean = np.mean(image, axis=(0, 1))
    rgb_std  = np.std(image,  axis=(0, 1))

    # ── HSV Mean & Std (6) ──────────────────────────────────────────────
    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std  = np.std(hsv,  axis=(0, 1))

    # ── Hu Moments (7) ──────────────────────────────────────────────────
    try:
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    except Exception:
        moments = np.zeros(7)

    # Stack 39 fitur: GLCM(20) + RGB(6) + HSV(6) + Hu(7)
    return np.hstack([glcm_features, rgb_mean, rgb_std, hsv_mean, hsv_std, moments])
