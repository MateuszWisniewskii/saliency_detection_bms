import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import load_and_preprocess_image
from src.bms_core import generate_boolean_maps, compute_mean_attention_map
from src.postprocessing import postprocess_saliency_map
from src import config


def process_image_pipeline(image_path: str, output_dir: str, use_lab: bool = True):
    mode_name = "Lab" if use_lab else "RGB"
    print(f"Rozpoczynam przetwarzanie: {image_path} (tryb: {mode_name})...")

    # KROK 1: Preprocessing
    if use_lab:
        lab_img = load_and_preprocess_image(image_path)
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")
        h, w = img.shape[:2]
        new_h = int(config.RESIZE_WIDTH * (h / w))
        img = cv2.resize(img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        # W trybie RGB używamy kanałów RGB zamiast Lab
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # KROK 2: Generowanie Map Boole'a
    boolean_maps = generate_boolean_maps(lab_img)
    print(f"Wygenerowano {len(boolean_maps)} map Boole'a.")

    # KROK 3: Wyznaczanie map uwagi (Flood Fill + normalizacja L2)
    mean_attention_map = compute_mean_attention_map(boolean_maps)

    # KROK 4: Postprocessing
    final_saliency_map = postprocess_saliency_map(mean_attention_map)

    # --- ZAPIS I WIZUALIZACJA ---
    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)

    # Wczytanie oryginału do wizualizacji (przeskalowany tak samo jak w preprocessingu)
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    new_h = int(config.RESIZE_WIDTH * (h / w))
    orig_img = cv2.resize(orig_img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Heatmap nałożony na oryginał
    heatmap = cv2.applyColorMap(final_saliency_map, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    # Normalizacja mean_attention_map do [0,255] tylko do celów wizualizacji
    mean_map_visual = cv2.normalize(
        mean_attention_map, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Wykres z 4 panelami
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"BMS Saliency Detection - tryb: {mode_name}", fontsize=14)

    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Oryginał")
    axes[0].axis('off')

    axes[1].imshow(mean_map_visual, cmap='gray')
    axes[1].set_title("Średnia mapa uwagi\n(przed post-processingiem)")
    axes[1].axis('off')

    axes[2].imshow(final_saliency_map, cmap='gray')
    axes[2].set_title("BMS Saliency Map\n(po post-processingiem)")
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Nałożony Heatmap")
    axes[3].axis('off')

    plt.tight_layout()

    # Zapis wyników
    raw_out_path = os.path.join(output_dir, f"{base_name}_{mode_name}_saliency.png")
    cv2.imwrite(raw_out_path, final_saliency_map)

    plot_out_path = os.path.join(output_dir, f"{base_name}_{mode_name}_plot.png")
    plt.savefig(plot_out_path, bbox_inches='tight')
    plt.close()

    print(f"Zapisano wyniki w: {output_dir}\n")


def compare_lab_vs_rgb(image_path: str, output_dir: str):
    """
    Uruchamia pipeline dla tego samego zdjęcia w trybie Lab i RGB,
    a następnie generuje jeden wykres porównawczy obok siebie.
    """
    print(f"Porównanie Lab vs RGB dla: {image_path}")

    # Lab
    lab_img = load_and_preprocess_image(image_path)
    boolean_maps_lab = generate_boolean_maps(lab_img)
    mean_map_lab = compute_mean_attention_map(boolean_maps_lab)
    saliency_lab = postprocess_saliency_map(mean_map_lab)

    # RGB
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_h = int(config.RESIZE_WIDTH * (h / w))
    img = cv2.resize(img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boolean_maps_rgb = generate_boolean_maps(rgb_img)
    mean_map_rgb = compute_mean_attention_map(boolean_maps_rgb)
    saliency_rgb = postprocess_saliency_map(mean_map_rgb)

    # Oryginał do wyświetlenia
    orig_img = cv2.imread(image_path)
    orig_img = cv2.resize(orig_img, (config.RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Wykres porównawczy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Porównanie przestrzeni barw: Lab vs RGB", fontsize=14)

    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Oryginał")
    axes[0].axis('off')

    axes[1].imshow(saliency_lab, cmap='gray')
    axes[1].set_title("BMS - CIE Lab\n(zgodnie z artykułem)")
    axes[1].axis('off')

    axes[2].imshow(saliency_rgb, cmap='gray')
    axes[2].set_title("BMS - RGB\n(baseline)")
    axes[2].axis('off')

    plt.tight_layout()

    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)
    compare_out_path = os.path.join(output_dir, f"{base_name}_lab_vs_rgb.png")
    plt.savefig(compare_out_path, bbox_inches='tight')
    plt.close()

    print(f"Zapisano porównanie: {compare_out_path}\n")


if __name__ == "__main__":
    INPUT_DIR = "data/inputs"
    OUTPUT_DIR = "data/outputs"

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_to_process = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_extensions)
    ]

    if not images_to_process:
        print(f"Wrzuć zdjęcia do folderu '{INPUT_DIR}' i uruchom ponownie!")
    else:
        for img_name in images_to_process:
            img_path = os.path.join(INPUT_DIR, img_name)

            # Pełny pipeline w trybie Lab (domyślny, zgodny z artykułem)
            process_image_pipeline(img_path, OUTPUT_DIR, use_lab=True)

            # Pełny pipeline w trybie RGB (do porównania)
            process_image_pipeline(img_path, OUTPUT_DIR, use_lab=False)

            # Wykres porównawczy Lab vs RGB obok siebie
            compare_lab_vs_rgb(img_path, OUTPUT_DIR)

        print("Zakończono przetwarzanie wszystkich obrazów!")