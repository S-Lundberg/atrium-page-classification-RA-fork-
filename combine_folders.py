import os
import shutil
import pandas as pd

# ============================================
# CONFIG
# ============================================

def merge_folders(
    folderA,
    folderB,
    output_root,
    csvA,
    csvB,
    excludeA=None,
    excludeB=None,
):
    """
    Slår ihop två mappar med identisk struktur.
    Tar bort dubbletter via filnamn.
    Exkluderar valfria subfolders.
    Slår ihop CSV-filer och tar bort dubbletter via local_filename.
    """

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    excludeA = set(excludeA or [])
    excludeB = set(excludeB or [])

    # --------------------------------------------
    # Helper: walk folder
    # --------------------------------------------
    def collect_images(root, exclude):
        images = []
        for path, dirs, files in os.walk(root):
            # hoppa över exkluderade subfolders
            rel_path = os.path.relpath(path, root)
            if rel_path != ".":
                top_folder = rel_path.split(os.sep)[0]
                if top_folder in exclude:
                    continue

            for f in files:
                if os.path.splitext(f.lower())[1] in exts:
                    full = os.path.join(path, f)
                    rel = os.path.relpath(full, root)
                    images.append((full, rel, f))
        return images

    # --------------------------------------------
    # Collect images
    # --------------------------------------------
    imgsA = collect_images(folderA, excludeA)
    imgsB = collect_images(folderB, excludeB)

    print(f"Folder A images: {len(imgsA)}")
    print(f"Folder B images: {len(imgsB)}")

    # --------------------------------------------
    # Remove duplicates by filename
    # --------------------------------------------
    seen = set()
    unique_images = []

    for full, rel, fname in imgsA + imgsB:
        if fname not in seen:
            seen.add(fname)
            unique_images.append((full, rel))

    print(f"Unique images (by filename): {len(unique_images)}")

    # --------------------------------------------
    # Copy unique images to output
    # --------------------------------------------
    for full, rel in unique_images:
        dst = os.path.join(output_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(full, dst)

    print("Images merged successfully.")

    # --------------------------------------------
    # Merge CSV files
    # --------------------------------------------
    dfA = pd.read_csv(csvA)
    dfB = pd.read_csv(csvB)

    df = pd.concat([dfA, dfB], ignore_index=True)

    if "local_filename" in df.columns:
        df = df.drop_duplicates(subset=["local_filename"], keep="first")

    merged_csv = os.path.join(output_root, "merged.csv")
    df.to_csv(merged_csv, index=False)

    print(f"Merged CSV saved to: {merged_csv}")
    print("Done.")


# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":

    # Snabbt att byta här:
    folderA = r"C:\RA-CLIP\Dataset_03_24"
    folderB = r"C:\RA-CLIP\New_test_set"
    output_root = r"C:\path\to\merged_test_set"

    csvA = folderA + "\\trying_images.csv"
    csvB = folderB + "\\New_test_set_images.csv"

    # Lägg till subfolders du vill hoppa över
    excludeA = []
    excludeB = ["LINE_HW","TEXT_HW"]

    merge_folders(
        folderA=folderA,
        folderB=folderB,
        output_root=output_root,
        csvA=csvA,
        csvB=csvB,
        excludeA=excludeA,
        excludeB=excludeB,
    )
