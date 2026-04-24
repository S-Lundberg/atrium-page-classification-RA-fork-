import pandas as pd
import os
import shutil


def validate_csv(df, filecol, label1, label2):
    dupes = df[df.duplicated(filecol, keep=False)]
    print("\n=== CSV VALIDATION ===")
    if len(dupes) > 0:
        print("⚠ Duplicates found:")
        print(dupes[[filecol, label1, label2]].to_string())
    else:
        print("✓ No duplicates")
    return dupes


def check_missing_files(df, image_folder, filecol):
    print("\n=== FILE CHECK ===")

    csv_files = set(df[filecol].tolist())
    folder_files = {
        f for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    }

    missing_in_folder = csv_files - folder_files
    missing_in_csv = folder_files - csv_files

    if missing_in_folder:
        print("⚠ Missing in folder:")
        for f in missing_in_folder:
            print("  -", f)
    else:
        print("✓ No missing files in folder")

    if missing_in_csv:
        print("⚠ Extra files in folder:")
        for f in missing_in_csv:
            print("  -", f)
    else:
        print("✓ No extra files in folder")

    return missing_in_folder, missing_in_csv


def sort_by_label(df, image_folder, label, filecol):
    print(f"\n=== SORTING BY {label} ===")
    moved = 0
    skipped = 0

    for _, row in df.iterrows():
        category = str(row[label])
        filename = row[filecol]

        src = os.path.join(image_folder, filename)
        if not os.path.exists(src):
            skipped += 1
            continue

        dest_dir = os.path.join(image_folder, category)
        os.makedirs(dest_dir, exist_ok=True)  # <-- skapar subfolder

        shutil.move(src, os.path.join(dest_dir, filename))
        moved += 1

    print(f"Moved: {moved}, Skipped: {skipped}")
    return moved, skipped


def sort_second_level(df, image_folder, label1, label2, filecol):
    print(f"\n=== SECOND-LEVEL SORTING BY {label2} ===")
    moved = 0
    skipped = 0

    for _, row in df.iterrows():
        cat1 = str(row[label1])
        cat2 = str(row[label2])
        filename = row[filecol]

        src = os.path.join(image_folder, cat1, filename)
        if not os.path.exists(src):
            skipped += 1
            continue

        dest_dir = os.path.join(image_folder, cat1, cat2)
        os.makedirs(dest_dir, exist_ok=True)  # <-- skapar sub-subfolder

        shutil.move(src, os.path.join(dest_dir, filename))
        moved += 1

    print(f"Moved: {moved}, Skipped: {skipped}")
    return moved, skipped


def run_pipeline(image_folder, csv_path, label1, label2, filecol):
    df = pd.read_csv(csv_path)

    validate_csv(df, filecol, label1, label2)
    check_missing_files(df, image_folder, filecol)
    sort_by_label(df, image_folder, label1, filecol)
    sort_second_level(df, image_folder, label1, label2, filecol)

    print("\n=== DONE ===")





