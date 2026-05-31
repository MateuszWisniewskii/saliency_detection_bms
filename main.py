import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importujemy nasze moduły z pakietu src
from src.preprocessing import load_and_preprocess_image
from src.bms_core import generate_boolean_maps, compute_mean_attention_map
from src.postprocessing import postprocess_saliency_map
from src import config

def process_image_pipeline(image_path: str, output_dir: str):
    print(f"Rozpoczynam przetwarzanie: {image_path}...")
    
    # KROK 1: Preprocessing
    lab_img = load_and_preprocess_image(image_path)
    
    # KROK 2: Generowanie Map Boole'a
    boolean_maps = generate_boolean_maps(lab_img)
    print(f"Wygenerowano {len(boolean_maps)} map Boole'a.")
    
    # KROK 3: Wyznaczanie rdzenia uwag (Flood Fill + L2)
    mean_attention_map = compute_mean_attention_map(boolean_maps)
    
    # KROK 4: Postprocessing (Saliency Map)
    final_saliency_map = postprocess_saliency_map(mean_attention_map)
    
    # --- ZAPIS I WIZUALIZACJA ---
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    
    # Zapis surowej mapy
    raw_out_path = os.path.join(output_dir, f"{base_name}_saliency.png")
    cv2.imwrite(raw_out_path, final_saliency_map)
    
    # Nałożenie mapy ciepła (heatmap) na oryginalny obraz do ładnej prezentacji
    # (Odczytujemy oryginał i skalujemy go tak samo jak w preprocessingu, by pasował wymiarami)
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    new_h = int(config.RESIZE_WIDTH * (h / w))
    orig_img = cv2.resize(orig_img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    
    heatmap = cv2.applyColorMap(final_saliency_map, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    # Wyświetlanie wyników w oknie
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Oryginał")
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("BMS Saliency Map")
    plt.imshow(final_saliency_map, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Nałożony Heatmap")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Zapis podglądu (plot)
    plot_out_path = os.path.join(output_dir, f"{base_name}_plot.png")
    plt.savefig(plot_out_path, bbox_inches='tight')
    print(f"Zapisano wyniki w: {output_dir}\n")
    
    # Odkomentuj poniższą linię, jeśli chcesz, żeby okienko z wykresem otwierało się na ekranie
    # plt.show()


if __name__ == "__main__":
    # Katalogi z danymi
    INPUT_DIR = "data/inputs"
    OUTPUT_DIR = "data/outputs"
    
    # Sprawdzamy wszystkie pliki w folderze wejściowym
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_to_process = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images_to_process:
        print(f"Wrzuć jakieś zdjęcia (np. .jpg lub .png) do folderu '{INPUT_DIR}' i uruchom skrypt ponownie!")
    else:
        for img_name in images_to_process:
            img_path = os.path.join(INPUT_DIR, img_name)
            process_image_pipeline(img_path, OUTPUT_DIR)
            
        print("🎉 Zakończono przetwarzanie wszystkich obrazów!")