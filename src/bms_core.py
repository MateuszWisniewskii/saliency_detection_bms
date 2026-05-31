import cv2
import numpy as np
from src import config

def generate_boolean_maps(lab_img: np.ndarray) -> list:
    """
    Generuje zestaw map Boole'a (binarnych) poprzez progowanie każdego kanału Lab,
    zgodnie z krokami zdefiniowanymi w configu.
    """
    h, w, channels = lab_img.shape
    # Jądro dla operacji morfologicznego otwarcia (redukcja szumów)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.OPENING_KERNEL_SIZE, config.OPENING_KERNEL_SIZE))
    
    boolean_maps = []
    
    # Przechodzimy przez kanały L, a oraz b
    for c in range(channels):
        channel_data = lab_img[:, :, c]
        
        # Próbkowanie progu (theta) od 0 do 255 co wartość delta (krok)
        for theta in range(0, 256, config.STEP_SIZE):
            # Tworzymy mapę binarną (piksele > theta stają się 255, reszta 0)
            _, b_map = cv2.threshold(channel_data, theta, 255, cv2.THRESH_BINARY)
            # Tworzymy zanegowaną wersję mapy
            _, b_inv = cv2.threshold(channel_data, theta, 255, cv2.THRESH_BINARY_INV)
            
            # Usuwamy drobny szum operacją otwarcia
            b_map_opened = cv2.morphologyEx(b_map, cv2.MORPH_OPEN, kernel)
            b_inv_opened = cv2.morphologyEx(b_inv, cv2.MORPH_OPEN, kernel)
            
            boolean_maps.append(b_map_opened)
            boolean_maps.append(b_inv_opened)
            
    return boolean_maps

def get_surrounded_regions(b_map: np.ndarray) -> np.ndarray:
    """
    Kluczowa funkcja algorytmu BMS. Wykorzystuje algorytm Flood Fill 
    do maskowania regionów stykających się z krawędziami obrazu.
    Zwraca mapę, na której regiony "otoczone" (wewnętrzne) mają wartość 1.
    """
    h, w = b_map.shape
    filled_map = b_map.copy()
    
    # Maska dla OpenCV floodFill musi być o 2 piksele większa niż obraz
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Krok 1: Wylewamy "wirtualną farbę" (o wartości 127) z każdej krawędzi obrazu.
    # Wartość 127 pozwala nam odróżnić wypełnione regiony krawędziowe od oryginalnych 0 i 255.
    
    # Górna i dolna krawędź
    for x in range(w):
        if filled_map[0, x] != 127:
            cv2.floodFill(filled_map, mask, (x, 0), 127)
        if filled_map[h - 1, x] != 127:
            cv2.floodFill(filled_map, mask, (x, h - 1), 127)
            
    # Lewa i prawa krawędź
    for y in range(h):
        if filled_map[y, 0] != 127:
            cv2.floodFill(filled_map, mask, (0, y), 127)
        if filled_map[y, w - 1] != 127:
            cv2.floodFill(filled_map, mask, (w - 1, y), 127)
            
    # Krok 2: Cokolwiek po zalaniu NIE ma wartości 127, oznacza że było zamknięte 
    # wewnątrz (otoczone) i woda z zewnątrz tam nie dotarła. Aktywujemy te regiony.
    attention_map = np.zeros_like(b_map, dtype=np.float32)
    attention_map[filled_map != 127] = 1.0
    
    return attention_map

def compute_mean_attention_map(boolean_maps: list) -> np.ndarray:
    """
    Oblicza mapy uwagi dla każdej mapy Boole'a, dylatuje, normalizuje i uśrednia.
    """
    if not boolean_maps:
        raise ValueError("Lista map Boole'a jest pusta!")
        
    h, w = boolean_maps[0].shape
    mean_a_map = np.zeros((h, w), dtype=np.float32)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.DILATION_KERNEL_1, config.DILATION_KERNEL_1))
    
    for b_map in boolean_maps:
        # Krok 1: Aktywacja regionów o zamkniętych konturach (surroundedness)
        a_map = get_surrounded_regions(b_map)
        
        # Krok 2: Dylatacja (karanie małych/rozrzuconych aktywnych obszarów - zgodnie z publikacją)
        a_map_dilated = cv2.dilate(a_map, dilate_kernel)
        
        # Krok 3: Normalizacja L2, czyli podzielenie wektora przez jego normę
        norm = np.linalg.norm(a_map_dilated)
        if norm > 0:
            a_map_normalized = a_map_dilated / norm
        else:
            a_map_normalized = a_map_dilated
            
        mean_a_map += a_map_normalized
        
    # Średnia arytmetyczna wszystkich znormalizowanych map
    mean_a_map /= len(boolean_maps)
    
    return mean_a_map