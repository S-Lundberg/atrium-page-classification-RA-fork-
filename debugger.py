import pandas as pd
import os
def debug_missing_prompts(metadata_csv, prompts_tsv):
    df_meta = pd.read_csv(metadata_csv)
    df_prompts = pd.read_csv(prompts_tsv, sep="\t")

    meta_files = set(df_meta["local_filename"])
    prompt_files = set(df_prompts["filename"])

    missing = sorted(list(meta_files - prompt_files))
    extra = sorted(list(prompt_files - meta_files))

    print("\n=== DEBUG: Missing prompt entries ===")
    print(f"Metadata rows: {len(meta_files)}")
    print(f"Prompt rows:   {len(prompt_files)}")
    print(f"Missing files: {len(missing)}")

    for f in missing:
        row = df_meta[df_meta["local_filename"] == f].iloc[0]
        print(f"- {f} | class={row['true_cat']} | metadata={row.to_dict()}")

    print("\n=== DEBUG: Extra prompt entries (should not happen) ===")
    for f in extra:
        print(f"- {f}")

    # Check prompt count per file
    print("\n=== DEBUG: Prompt count per file ===")
    counts = df_prompts.groupby("filename").size()
    wrong = counts[counts != counts.mode()[0]]

    if len(wrong) == 0:
        print("All files have correct number of prompts.")
    else:
        print("Files with incorrect number of prompts:")
        for fn, c in wrong.items():
            print(f"- {fn}: {c} prompts")

    return missing

def find_images_not_in_metadata(image_root, metadata_csv):
    # Load metadata
    df = pd.read_csv(metadata_csv)
    meta_files = set(df["local_filename"].astype(str))

    # Collect all image files recursively, with full relative paths
    image_paths = []
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, image_root)
                image_paths.append((f, rel_path))

    # Compare only by filename (metadata uses filenames, not paths)
    image_filenames = set([name for name, _ in image_paths])

    missing_filenames = sorted(list(image_filenames - meta_files))

    print("\n=== Images present in folder but missing in metadata ===")
    print(f"Total images in folder: {len(image_filenames)}")
    print(f"Total metadata entries: {len(meta_files)}")
    print(f"Missing images: {len(missing_filenames)}\n")

    # Print missing images with their subfolder paths
    for filename in missing_filenames:
        # find all paths where this filename occurs
        paths = [p for (name, p) in image_paths if name == filename]
        for p in paths:
            print(f"- {filename}  |  subfolder: {p}")

    return missing_filenames



if __name__ == "__main__":
	debug_missing_prompts(
    metadata_csv="C:/RA-CLIP/Dataset_03_24/trying_images.csv",
    prompts_tsv="./category_descriptions/prompts/dataset_prompts.tsv"
)
	find_images_not_in_metadata(
    image_root="C:/RA-CLIP/Dataset_03_24/", 
    metadata_csv="C:/RA-CLIP/Dataset_03_24/trying_images.csv"
)

