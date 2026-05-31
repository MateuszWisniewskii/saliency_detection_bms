import cv2
import numpy as np
from src import config

def postprocess_saliency_map(mean_a_map: np.ndarray) -> np.ndarray:
    """
    Przetwarza uśrednioną mapę uwagi do postaci końcowej mapy Saliency
    (pod kątem predykcji fiksacji wzroku).
    """
    # 1. Druga dylatacja (poszerzenie obszarów przed rozmyciem)
    dilate_kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (config.DILATION_KERNEL_2, config.DILATION_KERNEL_2))
    saliency_map = cv2.dilate(mean_a_map, dilate_kernel_2)
    
    # 2. Wygładzenie Gaussa
    # Używamy (0,0) dla rozmiaru jądra, co pozwala OpenCV automatycznie
    # wyliczyć optymalny rozmiar na podstawie odchylenia standardowego sigmaX (nasze GAUSSIAN_BLUR_STD).
    saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigmaX=config.GAUSSIAN_BLUR_STD)
    
    # 3. Normalizacja do zakresu [0, 255] (ułatwia zapis i wyświetlanie jako obraz)
    saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return saliency_map