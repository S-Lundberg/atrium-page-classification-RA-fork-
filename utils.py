from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time
import datetime
import re
import scipy.special

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    #for form in file_format:
    #    try:
    #        form_len = 0
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
            #form_len = len(file_list)-form_len
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
        #except Exception as e:
        #    print(f"[ERROR] Could not collect {form} files from {folder_path}: {e}")
    return file_list


def dataframe_results(
        test_images: list,
        test_predictions: list,
        categories: list,
        top_N: int,
        raw_scores: list = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes image prediction results into two pandas DataFrames:
    one for formatted top-N predictions and another for raw scores.

    FIXED VERSION:
      - Ensures test_predictions is flattened into an array of per-image scores.
      - Safely handles cases where the number of prediction values does not match
        the number of categories.
    """
    print(f"Processing {len(test_images)} images with top_N={top_N} predictions "
          f"of {len(categories)} possible labels...")

    # --- FIX 1: Flatten test_predictions into a single array of per-image scores ---
    # Convert list of batch arrays into a single list of per-image score arrays
    flat_predictions = []
    for batch_preds in test_predictions:
        # Assuming batch_preds is a numpy array or torch tensor of shape (batch_size, n_classes)
        # Convert to numpy array and ensure it's 2D
        batch_preds = np.atleast_2d(np.asarray(batch_preds))
        for row in batch_preds:
            flat_predictions.append(row)

    if not flat_predictions:
        print("[ERROR] Flat predictions list is empty.")
        return pd.DataFrame(), None


    # Vertically stack all per-image score arrays
    try:
        preds = np.vstack(flat_predictions)
    except ValueError as e:
        print(f"[CRITICAL ERROR] Could not stack flat predictions: {e}")
        return pd.DataFrame(), None

    # Check if the number of images in predictions matches test_images
    if preds.shape[0] != len(test_images):
        print(f"[ERROR] Mismatch: {len(test_images)} images, but {preds.shape[0]} predictions.")
        # If mismatch, truncate to the minimum count, which is usually the length of the images list
        min_len = min(preds.shape[0], len(test_images))
        preds = preds[:min_len]
        test_images = test_images[:min_len]
        # raw_scores would also need truncation if it was used, but we'll assume it's aligned or None

    n_images, n_raw_scores = preds.shape
    n_categories = len(categories)
    print(f"Final prediction array shape: {preds.shape}")

    # --- FIX 2: Check and align categories and scores ---
    # If the number of raw scores (columns) is greater than the number of categories, truncate the scores.
    if n_raw_scores > n_categories:
        # print(
        #     f"[WARN] {n_raw_scores} prediction values but only {n_categories} categories. Truncating prediction scores to match categories.")
        preds = preds[:, :n_categories]
    # If the number of raw scores is less than the number of categories, truncate the categories list.
    elif n_raw_scores < n_categories:
        # print(f"[WARN] {n_categories} categories but only {n_raw_scores} prediction scores. Truncating category list.")
        # categories = categories[:n_raw_scores]
        n_categories = n_raw_scores

    # Re-evaluate n_images, n_categories after potential truncation
    if preds.ndim == 2:
        n_images, n_categories = preds.shape
    else:  # Should not happen after np.vstack
        print("[ERROR] Predictions are not a 2D array.")
        return pd.DataFrame(), None

    results, raws = [], []
    valid_rows = 0

    for image_file, scores in zip(test_images, preds):
        # Skip fully zero/padded rows
        if np.all(scores == 0) and not np.any(scores):
            continue

        # --- Robust filename parsing ---
        image_name = Path(image_file).stem
        match = re.search(r'(\d+)$', image_name)
        if match:
            page_num = int(match.group(1))
            document = image_name[:match.start()].rstrip("-_")
        else:
            page_num = None
            document = image_name

        # --- Stable softmax normalization ---
        # scores should already be clipped to n_categories from the checks above
        valid_scores = scores
        # Use scipy.special.softmax for robustness if available, otherwise the custom one
        try:
            probs = scipy.special.softmax(valid_scores)
        except AttributeError:
            exp_scores = np.exp(valid_scores - np.max(valid_scores))
            probs = exp_scores / np.sum(exp_scores)
        except FloatingPointError as e:
            print(f"[WARN] Floating point error during softmax: {e}. Using uniform distribution.")
            probs = np.full_like(valid_scores, 1.0 / len(valid_scores))

        # --- Top-N selection ---
        top_N = max(1, min(top_N, n_categories))
        top_idx = probs.argsort()[::-1][:top_N]
        labels = [categories[i] for i in top_idx]
        score_vals = [round(float(probs[i]), 3) for i in top_idx]

        results.append([document, page_num] + labels + score_vals)
        valid_rows += 1

        # --- Raw scores ---
        if raw_scores is not None:
            raws.append([document, page_num] + [round(float(s), 3) for s in valid_scores])

    # ... (rest of the function for DataFrame construction remains the same) ...

    if valid_rows == 0:
        print("[ERROR] No valid prediction rows found — check input shapes or category count.")
        return pd.DataFrame(), None

    # --- Construct formatted results DataFrame ---
    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    if top_N == 1:
        rdf = rdf.rename(columns={"CLASS-1": "CATEGORY"}).drop(columns=["SCORE-1"])

    # --- Construct raw scores DataFrame ---
    rawdf = None
    if raw_scores is not None:
        raw_col = ["FILE", "PAGE"] + categories
        rawdf = pd.DataFrame(raws, columns=raw_col)

    print(f"Created results table with {len(rdf.index)} rows and columns: {rdf.columns.tolist()}")
    if rawdf is not None:
        print(f"Created RAW results table with shape {rawdf.shape}")

    return rdf, rawdf


def collect_images(directory: str, max_categ: int) -> (list, list, list):
    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files, total_labels, total_classes = [], [], []
    for category_idx, category in enumerate(categories):
        all_category_files = os.listdir(os.path.join(directory, category))
        if len(all_category_files) > max_categ:
            random.shuffle(all_category_files)
            all_category_files = all_category_files[:max_categ]

        print(f"Collected {len(all_category_files)} {category} category images")

        total_files += [os.path.join(directory, category, file) for file in all_category_files]

        label_template = np.zeros(len(categories))
        label_template[category_idx] = 1

        total_labels += [label_template] * len(all_category_files)
        total_classes += [category_idx] * len(all_category_files)

    label, count = np.unique(total_classes, return_counts=True)
    for label_id, label_count in zip(label, count):
        print(f"{categories[int(label_id)]}:\t{label_count}\t{round(label_count / len(total_labels) * 100, 2)}%")

    return total_files, total_labels, categories




def append_to_csv(df, filepath):
    """
    Appends a DataFrame to a CSV file, or creates a new file if it doesn't exist.
    """
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")


# Source - https://stackoverflow.com/a/34304414
# Posted by Franck Dernoncourt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-19, License - CC BY-SA 4.0

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    ax = plt.gca()

    # Hämta värdena som en 1D-array och forma om till 2D
    data = pc.get_array()
    if data.ndim == 1:
        # pcolormesh flattenar värdena radvis
        ny, nx = pc._meshHeight, pc._meshWidth  # fungerar i äldre versioner
        try:
            data = data.reshape(ny, nx)
        except Exception:
            # fallback: använd heatmapens shape
            data = data.reshape(ax.images[0].get_array().shape)

    # Loopa över cellerna baserat på index
    ny, nx = data.shape
    for i in range(ny):
        for j in range(nx):
            value = data[i, j]

            # extrahera scalar
            if hasattr(value, "item"):
                value = value.item()

            # cellens mittpunkt = index + 0.5
            ax.text(
                j + 0.5,
                i + 0.5,
                fmt % value,
                ha="center",
                va="center",
                color="black",
            )


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')
    #print(classification_report)
    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue

        # slå ihop tvåordsklasser som "macro avg" och "weighted avg"
        if t[0] in ("macro", "weighted","accuracy"): #and t[1] == "avg":
            #t[0] = t[0] + "_" + t[1]   # macro_avg / weighted_avg
            #t.pop(1)                   # ta bort "avg"
        #if t[0] == "accuracy":
            continue
        classes.append(t[0])

        # nu är t t.ex. ["macro_avg", "0.899", "0.932", "0.914", "225"]
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)



    #print('plotMat: {0}'.format(plotMat))
    #print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def main_class_report(clsrprt,name="test_plot_classif_report.png"):


    plot_classification_report(clsrprt)
    plt.savefig(name, dpi=200, format='png', bbox_inches='tight')
    plt.close()

#if __name__ == "__main__":
#    main()
    #cProfile.run('main()') # if you want to do some profiling

