import pandas as pd
import os
from pathlib import Path
import clip
import tiktoken
import torch
PROMPT_TEMPLATES = {
    #"LINE_MIX": "En arkivsida i tabell- eller formulärlayout med blandade textstilar, inklusive handskriven, tryckt och maskinskriven text {metadata}.",
    "LINE_MIX": "En arkivsida i tabell- eller formulärlayout med blandade textstilar {metadata}.",
    "MISC": "En arkivsida med avvikande innehåll, såsom omslag, titelblad eller annat material utanför huvudtexten {metadata}.",
    #"DRAW": "En arkivsida som innehåller en ritning, karta, skiss eller schematisk illustration, ibland placerad i tabell- eller formulärlayout {metadata}.",
    "DRAW": "En arkivsida som innehåller en ritning, karta, skiss eller schematisk illustration {metadata}.",
    "LINE_TP": "En arkivsida med tryckt eller maskinskriven text organiserad i tabell- eller formulärstruktur {metadata}.",
   # "PHOTO": "En arkivsida som innehåller ett fotografi eller fotografiskt motiv, ibland placerat i tabell- eller formulärlayout {metadata}.",
    "PHOTO": "En arkivsida som innehåller ett fotografi eller fotografiskt motiv {metadata}.",
    "TEXT_Tp": "En arkivsida med löpande tryckt eller maskinskriven text i stycken {metadata}.",
    "LINE_HW": "En arkivsida med handskriven text organiserad i tabell- eller formulärstruktur {metadata}.",
    "TEXT": "En arkivsida med löpande text bestående av blandade stilar, såsom handskriven, tryckt och maskinskriven text {metadata}.",
    "TEXT_HW": "En arkivsida med löpande handskriven text i stycken {metadata}."
}


CLIP_MAX_TOKENS = 65  # CLIP standard

def smart_truncate_text(text: str, max_chars: int = 300) -> str:
    """Bevara början och slutet av texten vid trunkering."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-(max_chars // 2) :].lstrip()
    return head + " ... " + tail

def tokenize_safe_single(prompt: str, max_tokens: int = CLIP_MAX_TOKENS, max_attempts: int = 5):
    """
    Försök tokenisera en prompt. Om clip.tokenize kastar pga för lång input,
    trunkera smart och försök igen. Returnerar (tokens_tensor, was_truncated, final_prompt).
    """
    attempt = 0
    current = prompt
    was_truncated = False

    while attempt < max_attempts:
        try:
            tokens = clip.tokenize([current])  # kan kasta RuntimeError om för lång
            if tokens.shape[1] <= max_tokens:
                return tokens, was_truncated, current
            # Om tokeniseraren ändå returnerar längre än max_tokens, markera trunc och trunkera textnivå
            was_truncated = True
            current = smart_truncate_text(current, max_chars=max(100, int(len(current) * 0.5)))
            if not current.strip():
                print("EMPTY PROMPT AFTER TRUNCATION:", prompt)

        except RuntimeError:
            # clip.tokenize kan kasta; trunkera textnivå och försök igen
            was_truncated = True
            current = smart_truncate_text(current, max_chars=max(100, int(len(current) * 0.5)))
        attempt += 1

    # Sista utväg: hård trunkering till teckenlängd
    final = (prompt[:200] + " ... " + prompt[-20:]) if len(prompt) > 240 else prompt
    tokens = clip.tokenize([final])
    return tokens, True, final

def create_prompt(metadata_csv_path, category_tsv_path=None, out_path=None):
    """
    Loads metadata and class prompt templates, generates text prompts for each image,
    ensures all prompts are CLIP-safe using tiktoken (max 77 tokens),
    and writes dataset_prompts.tsv with:
        filename, class, prompt, prompt_truncated, prompt_preview

    Includes extensive debug output to detect rows that do not generate
    the expected number of prompts.
    """

    import pandas as pd
    from pathlib import Path
    import tiktoken

    print(f"Loading metadata CSV from: {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    # Initialize tiktoken encoder (GPT-2 BPE, same tokenization family as CLIP)
    clip_encoder = tiktoken.get_encoding("gpt2")
    CLIP_MAX_TOKENS = 65

    def truncate_to_clip(text: str, max_tokens=CLIP_MAX_TOKENS):
        """Truncate a string so that its token length does not exceed CLIP's limit,
        keeping the END of the text (metadata, dates, archive info)."""

        tokens = clip_encoder.encode(text)
        n = len(tokens)

        if n <= max_tokens:
            return text, False

        # how many tokens must be removed
        diff = n - max_tokens

        #print(f"Prompt exceeds {max_tokens} tokens ({len(tokens)}), truncating: '{text}'")

        # keep the last max_tokens tokens
        kept_tokens = tokens[diff:]
        truncated = clip_encoder.decode(kept_tokens)

        #print(f"truncated prompt: '{truncated}'")

        return truncated, True

    truncs = 0
    # Load class templates
    if category_tsv_path:
        print(f"Loading category prompt templates from: {category_tsv_path}")
        df_cat = pd.read_csv(category_tsv_path, sep=",")
        templates_by_class = df_cat.groupby("label")["description"].apply(list).to_dict()
    else:
        print("Using built-in PROMPT_TEMPLATES...")
        templates_by_class = PROMPT_TEMPLATES

    prompts_out = []
    rows = df.to_dict(orient="records")
    print(f"Processing {len(rows)} rows...")

    for row in rows:
        filename = row.get("local_filename", "<unknown>")
        klass = row["true_cat"]

        # Build metadata string
        metadata = build_metadata(row)
        if not metadata or not metadata.strip():
            print(f"[DEBUG] EMPTY METADATA for file: {filename}")
            metadata = "unknown source"

        # Get templates for this class
        templates = templates_by_class.get(klass)
        if not templates:
            print(f"[DEBUG] NO TEMPLATES FOUND for class '{klass}' in file: {filename}")
            templates = ["An archival page."]

        expected_templates = len(templates)
        generated_for_row = 0

        # Debug info container
        debug_info = {
            "filename": filename,
            "class": klass,
            "metadata": metadata,
            "expected_templates": expected_templates,
            "templates": templates,
            "empty_prompts": [],
            "format_errors": [],
            "truncated_prompts": 0,
            "final_prompts": []
        }

        for template in templates:
            # Build the raw prompt
            try:
                if "{metadata}" in template:
                    prompt = template.format(metadata=metadata)
                else:
                    prompt = f"{template.strip()} from {metadata}"
            except Exception as e:
                debug_info["format_errors"].append((template, str(e)))
                prompt = f"{klass}: archival page from {metadata}"

            if not prompt.strip():
                debug_info["empty_prompts"].append(template)
                prompt = f"{klass}: archival page without metadata"

            # Truncate using tiktoken BEFORE training
            final_prompt, was_truncated = truncate_to_clip(prompt)
            if was_truncated:
                debug_info["truncated_prompts"] += 1

            debug_info["final_prompts"].append(final_prompt)

            prompts_out.append({
                "filename": filename,
                "class": klass,
                "prompt": final_prompt,
                "prompt_truncated": was_truncated,
                "prompt_preview": (
                    final_prompt if len(final_prompt) <= 300
                    else final_prompt[:200] + "..." + final_prompt[-20:]
                )
            })

            generated_for_row += 1
        if was_truncated:
            truncs +=1
        #    print("\n=== DEBUG: ROW DID NOT GENERATE EXPECTED NUMBER OF PROMPTS ===")
        #    print(f"File: {filename}")
        #    print(f"Class: {klass}")
        #    print(f"Expected: {expected_templates}, Generated: {generated_for_row}")
        #    print("Metadata:", metadata)
        #    print("Templates:", templates)
        #    print("Empty prompts:", debug_info["empty_prompts"])
        #    print("Format errors:", debug_info["format_errors"])
        #    print("Truncated prompts:", debug_info["truncated_prompts"])
        #    print("Final prompts:", debug_info["final_prompts"])
        #print("================================================================\n")

    # Save output TSV
    out_path = out_path or Path("category_descriptions/prompts/")
    out_path.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(prompts_out)
    out_file = "test_set_prompts.tsv"
    out_df.to_csv(out_path / out_file, sep="\t", index=False)
    print(f"Truncated prompts: {truncs} out of {len(prompts_out)} total prompts.")
    print(f"Done! Created {out_file} with {len(prompts_out)} prompts.")





def safe_token_len(tokens):
    if tokens.dim() == 2:
        return tokens.shape[1]
    if tokens.dim() == 1:
        return tokens.shape[0]
    return CLIP_MAX_TOKENS + 1


def build_metadata(row):
    parts = []
    from difflib import SequenceMatcher

    if "serie" in row and pd.notna(row["serie"]):
        serie = str(row["serie"]).strip()
        arkiv = str(row["arkiv"]).strip()

        # Beräkna likhet
        similarity = SequenceMatcher(None, arkiv.lower(), serie.lower()).ratio()

        # Om arkiv och serie är "nästan samma"
        if similarity > 0.55:
            # behåll bara serien
            parts.append(serie)
        else:
            # behåll båda
            parts.append(f"från {arkiv}, {serie}")

    if "datering" in row and pd.notna(row["datering"]):
        parts.append(f"dated: {row['datering']}")
    return ", ".join(parts) if parts else "."

def create_prompt_old(csv_path):
    df = pd.read_csv(csv_path) 
    prompts = []
    for _, row in df.iterrows():
        klass = row["true_cat"]
        metadata = build_metadata(row)
        template = PROMPT_TEMPLATES.get(klass, "En arkivsida. Metadata: {metadata}.")
        prompt = template.format(metadata=metadata)
        if len(prompt)>250:
            print(f"Prompt for {row['local_filename']} exceeds 300 characters, truncating.")
            prompt = prompt[:200] + "..." + prompt[-20:]
        prompts.append({
            "filename": row["local_filename"],
            "class": klass,
            "prompt": prompt
        })

    out_path = "category_descriptions\\prompts\\"
    out_df = pd.DataFrame(prompts)
    out_df.to_csv(out_path + "test_set_prompts.tsv", sep="\t", index=False)

    print("Klar! prompts.tsv skapad.")


#if "__main__" == __name__:
#    create_prompt(r"C:\RA-CLIP\Dataset_03_24\trying_images.csv")