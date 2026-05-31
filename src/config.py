"""
Konfiguracja hiperparametrów algorytmu BMS na podstawie publikacji z ICCV 2013.
"""

# --- Parametry wejściowe ---
RESIZE_WIDTH = 600        # Szerokość, do której skalowany jest obraz wejściowy 

# --- Parametry generowania map Boole'a ---
STEP_SIZE = 8             # Krok próbkowania (delta) progowania dla kanałów Lab

# --- Parametry morfologiczne ---
OPENING_KERNEL_SIZE = 5   # omega_0: Rozmiar jądra operacji otwarcia (redukcja szumów) 
DILATION_KERNEL_1 = 7     # omega_d1: Rozmiar jądra dylatacji (przed normalizacją) 
DILATION_KERNEL_2 = 23    # omega_d2: Rozmiar jądra dylatacji (przed ostatecznym rozmyciem) 

# --- Parametry wygładzania końcowego ---
GAUSSIAN_BLUR_STD = 5    # sigma: Odchylenie standardowe rozmycia Gaussa 