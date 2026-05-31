import cv2
import numpy as np
from src import config

def load_and_preprocess_image(filepath: str) -> np.ndarray:
    """
    Wczytuje obraz z dysku, skaluje do stałej szerokości z zachowaniem 
    proporcji i konwertuje do przestrzeni barw CIE Lab.
    """
    # 1. Wczytanie obrazu
    # OpenCV domyślnie wczytuje w formacie BGR, a nie RGB!
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu ze ścieżki: {filepath}. Sprawdź, czy plik istnieje.")

    # 2. Skalowanie obrazu z zachowaniem proporcji (Aspect Ratio)
    h, w = img.shape[:2]
    if w != config.RESIZE_WIDTH:
        aspect_ratio = h / w
        new_h = int(config.RESIZE_WIDTH * aspect_ratio)
        img = cv2.resize(img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # 3. Konwersja do przestrzeni CIE Lab
    # Zgodnie z założeniami autorów (Sekcja 3.1) używamy przestrzeni Lab, 
    # ponieważ metryka odległości w niej dobrze odzwierciedla wizualne różnice kolorów dla człowieka.
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    return lab_img