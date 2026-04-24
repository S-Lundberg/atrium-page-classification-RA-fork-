import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os

def count_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    total = 0

    for path, dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                total += 1

    return total

folder = r"C:\path\to\your\folder"
print("Antal bilder:", count_images(folder))



CLASS_MAP = {
    "0": "DRAW",
    "1": "LINE_MIX",
    "2": "LINE_HW",
    "3": "LINE_TP",
    "4": "PHOTO",
    "5": "TEXT",
    "6": "TEXT_HW",
    "7": "TEXT_TP",
    "8": "MISC",
}

def on_key(event):
    global current_choice
    if event.key in CLASS_MAP.keys() or event.key in ["n", "b", "q"]:
        current_choice = event.key
        plt.close()


def annotate_csv(csv_path,csv_out_path,image_folder):

    # Skapa CSV om den saknas
    create_csv_if_missing(csv_path, image_folder)

    # Bygg index över alla bilder i subfolders
    image_index = build_image_index(image_folder)

    df = pd.read_csv(csv_path)
    print("Totalt antal rader i CSV:", len(df))
    print("Totalt antal bilder i folder:", count_images(image_folder))
    # Se till att kolumnen 'true' finns
    if "true" not in df.columns:
        df["true"] = -1
    else:
        df["true"] = df["true"].fillna(-1)
    # Klasslista
    CLASS_MAP = {
        "1": "DRAW",
        "2": "LINE_MIX",
        "3": "LINE_HW",
        "4": "LINE_TP",
        "5": "PHOTO",
        "6": "TEXT",
        "7": "TEXT_HW",
        "8": "TEXT_TP",
        "9": "MISC",
    }

    # Starta på första rad med true = -1
    start_index = df.index[df["true"] == -1]
    if len(start_index) == 0:
        df["true_cat"] = df["true"].astype(int).astype(str).map(CLASS_MAP)
        df.to_csv(csv_path,index=False)
        print(f"Alla {len(df)} rader är redan annoterade.")
        return
    index = start_index[0]

    current_choice = None

    def on_key(event):
        nonlocal current_choice
        if event.key in CLASS_MAP.keys() or event.key in ["n", "b", "q","s"]:
            current_choice = event.key
            plt.close()

    while 0 <= index < len(df):

        row = df.iloc[index]
        filename = row["local_filename"]

        img_path = image_index.get(filename, None)
        if not os.path.exists(img_path):
            print(f"Bild saknas: {filename}, hoppar över.")
            index += 1
            continue

        print(img_path)
        # Visa bild
        img = Image.open(img_path)
        current_choice = None

        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(18, 10))
        fig.canvas.mpl_connect("key_press_event", on_key)

        # --- Bild ---
        ax_img.imshow(img)
        ax_img.axis("off")

        # --- Textpanel ---
        ax_text.axis("off")
        class_text = "\n".join([f"{k} = {v}" for k, v in CLASS_MAP.items()])
        true_display = CLASS_MAP[str(int(row["true"]))] if int(row["true"]) != -1 else "-1"

        text_block = (
            f"Rad {index+1}/{len(df)}\n"
            f"Fil: {filename}\n"
            f"Nuvarande TRUE: {true_display}\n\n"
            f"KLASSER:\n{class_text}\n\n"
            f"[n = Nästa]\n"
            f"[b = Backa]\n"
            f"[q = Avsluta]"
        )

        ax_text.text(
            0.01, 0.99,
            text_block,
            va="top",
            ha="left",
            fontsize=16,
            family="monospace"
        )
        plt.tight_layout()
        plt.show()

        if current_choice is None:
            continue

        # Hantera val
        if current_choice in CLASS_MAP:
            df.loc[index, "true"] = int(current_choice)
            df.to_csv(csv_path, index=False)
            index += 1

        elif current_choice == "n":
            index += 1

        elif current_choice == "b":
            index -= 1
        
        elif current_choice == "s":
            index = 0

        elif current_choice == "q":
            print("Avslutar och sparar...")
            df.to_csv(csv_path, index=False)
            break

        else:
            fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(18, 10))
            fig.canvas.mpl_connect("key_press_event", on_key)

            # --- Bild ---
            ax_img.imshow(img)
            ax_img.axis("off")

            # --- Textpanel ---
            ax_text.axis("off")

            class_text = "\n".join([f"{k} = {v}" for k, v in CLASS_MAP.items()])

            text_block = (
                f"Rad {index+1}/{len(df)}\n"
                f"Fil: {filename}\n"
                f"Nuvarande TRUE: {true_display}\n\n"
                f"KLASSER:\n{class_text}\n\n"
                f"[n = Nästa]\n"
                f"[b = Backa]\n"
                f"[q = Avsluta]"
            )

            ax_text.text(
                0.01, 0.99,
                text_block,
                va="top",
                ha="left",
                fontsize=16,
                family="monospace"
            )
            plt.xlabel("Fel knapp? Prova igen..")
            plt.tight_layout()
            plt.show()
            continue

    df["true_cat"] = df["true"].astype(int).astype(str).map(CLASS_MAP)
    #df.loc[df["true"] == 2, "true"] = 3
    #df.loc[df["true_cat"] == "LINE_MIX", "true_cat"] = "LINE_HW"
    df.to_csv(csv_path,index=False)
    print("Klart!")
    

import os
import pandas as pd

def create_csv_if_missing(csv_path, image_root):
    """
    Skapar en CSV med local_filename + true = -1 om filen inte finns.
    Går rekursivt genom alla subfolders i image_root.
    """
    if os.path.exists(csv_path):
        print(f"CSV finns redan: {csv_path}")
        return

    print("CSV saknas – skapar ny...")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    rows = []

    for path, dirs, files in os.walk(image_root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                rows.append({"local_filename": f, "true": -1})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Ny CSV skapad med {len(df)} bilder.")

def build_image_index(image_root):
    """
    Returnerar en dict:
    { "filnamn.jpg": "C:/path/to/subfolder/filnamn.jpg" }
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    index = {}

    for path, dirs, files in os.walk(image_root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                index[f] = os.path.join(path, f)

    return index


if __name__ == "__main__":

    path="anno"
    out_path = "anno"
    csv_path = path + "\\anno_images.csv"
    #path = r"C:\RA_ufal_atrium\images_page\all_images"
    #csv_path = path + "\\anno_page.csv"
    #out_path = path
    
    annotate_csv(csv_path=csv_path,image_folder=path,csv_out_path=out_path)