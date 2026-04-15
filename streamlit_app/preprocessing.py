import cv2
import numpy as np

# ─── Preprocessing constants ────────────────────────────────────────────────
# These values are validated in the thesis research pipeline.
# Change only when re-validating the full feature-extraction pipeline.
IMAGE_SIZE        = (224, 224)          # Target width × height after resize
GAUSSIAN_KERNEL   = (5, 5)              # Kernel size for Gaussian blur
MORPH_KERNEL_SIZE = (5, 5)              # Kernel size for morphological closing
# HSV green range (H: 25-75°, S: 40-255, V: 50-255) tuned for weed segmentation
HSV_GREEN_LOWER   = np.array([25, 40,  50])
HSV_GREEN_UPPER   = np.array([75, 255, 255])
# ────────────────────────────────────────────────────────────────────────────


def preprocess_image_with_steps(img_path=None, image_bytes=None):
    """
    Run the full thesis preprocessing pipeline and return all intermediate images.
    Pipeline: Resize(224x224) -> Gaussian Blur -> HSV Thresholding -> Morphology Closing

    Returns a dict with RGB images ready for st.image():
        - 'original'   : resized input image (RGB)
        - 'hsv_mask'   : binary mask after HSV thresholding (3-channel grayscale)
        - 'segmented'  : final segmented image after morphology closing (RGB)
        - 'segmented_bgr': same but in BGR for OpenCV-based feature extraction
    Returns None if the image cannot be decoded.
    """
    if img_path:
        img = cv2.imread(img_path)
    elif image_bytes:
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Either img_path or image_bytes must be provided.")

    if img is None:
        return None

    # Step 1: Resize
    img_resized = cv2.resize(img, IMAGE_SIZE)

    # Step 2: Gaussian Blur
    blurred = cv2.GaussianBlur(img_resized, GAUSSIAN_KERNEL, 0)

    # Step 3: HSV Thresholding — green weed color range from thesis
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)

    # Step 4: Morphology Closing — fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to get segmented image
    segmented_bgr = np.zeros_like(img_resized, dtype=np.uint8)
    segmented_bgr[mask_closed > 0] = img_resized[mask_closed > 0]

    return {
        'original': cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
        'hsv_mask': cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2RGB),
        'segmented': cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB),
        'segmented_bgr': segmented_bgr,
    }


def preprocess_image(img_path=None, image_bytes=None):
    """
    Convenience wrapper — returns only the final segmented image (BGR) for feature extraction.
    """
    steps = preprocess_image_with_steps(img_path=img_path, image_bytes=image_bytes)
    if steps is None:
        return None
    return steps['segmented_bgr']
