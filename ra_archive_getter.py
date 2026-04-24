"""Get images/metadata from a Riksarkivet archive/volume/series."""

import requests
from pathlib import Path
import json
import os
import re
import zipfile
import random
import urllib.parse
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import datetime
# ============================================================
# RecordsAPI – search, JSON-LD, conditions, metadata
# ============================================================

def folding(items,ins=""):
    try:
        for i,j in items.items():

            if isinstance(j,dict) or isinstance(j,list):
                print(ins,i,": ")
                folding(j,"\t")

            else:
                print(ins,i,": ",j)
    except:
        for i,j in enumerate(items):

            if isinstance(j,dict) or isinstance(j,list):
                print(ins,i,": ")
                folding(j,"\t")

            else:
                print(ins,i,": ",j)


class RecordsAPI:
    BASE_URL = "https://data.riksarkivet.se/api/records"
    def __init__(self,params:dict=None) -> None:
        self.params = params
    def riksarkivet_records(self,
        text=None,
        name=None,
        from_year=None,
        to_year=None,
        only_dig=True
    ):
        """
        Anropar Riksarkivets records-API.
        Returnerar JSON-data.
        """
        # Lägg bara till parametrar som faktiskt används
        if text:
            self.params["text"] = text
        if name:
            self.params["name"] = name
        if from_year:
            self.params["year_min"] = from_year
        if to_year:
            self.params["year_max"] = to_year
        if only_dig:
            self.params["only_digitised_materials"]= True

        r = requests.get(self.BASE_URL, params=self.params, timeout=20)
        print(f"Search url: {r.url}")
        r.raise_for_status()
        return r.json()

    def get_items(self,
        text=None,
        name=None,
        from_year=None,
        to_year=None,
        only_dig=True
    ):
        hits = self.riksarkivet_records(text=text,
        name=name,
        from_year=from_year,
        to_year=to_year,
        only_dig=only_dig
    )
        return hits.get("items",[])
        
# ============================================================
# IIIFClient – collection → manifest → canvas → image
# ============================================================

class IIIFClient:
    URL_BASE = "https://lbiiif.riksarkivet.se/collection/arkiv/"

    def _download_file(self, url, file_path):
        try:
            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False

    def _extract_manifest_metadata(self, manifest):
        md = {
            "arkiv": None,
            "serie": None,
            "referenskod": None,
            "datering": None,
            "anmärkning": None,
        }

        for entry in manifest.get("metadata", []):
            # Hämta label robust
            label = None
            if isinstance(entry.get("label"), dict):
                # ta första språket som finns
                label = next(iter(entry["label"].values()))[0]
            elif isinstance(entry.get("label"), str):
                label = entry["label"]

            # Hämta value robust
            value = self._safe_value(entry)

            if label == "Arkiv":
                md["arkiv"] = value
            elif label == "Serie":
                md["serie"] = value
            elif label == "Referenskod":
                md["referenskod"] = value
            elif label == "Datering":
                md["datering"] = value
            elif label == "Anmärkning":
                md["anmärkning"] = value

        return md

    
    def _safe_get_json(self, url, pid=None, log_path=None):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️ Kunde inte läsa JSON från {url}: {e}")
            if log_path and pid:
                self._log_broken_manifest(url, pid, e, log_path)
            return None


    def _process_manifest(self, manifest, pid, manifest_index, folder, 
                        max_per_manifest, ead_meta):

        rows = []
        images_downloaded = 0

        # Manifest-level metadata (fallback)
        manifest_meta = self._extract_manifest_metadata(manifest)

        # Canvases
        canvases = manifest.get("items", [])
        #folding(canvases)
        if len(canvases) > max_per_manifest:
            canvases = random.sample(canvases, max_per_manifest)

        for canvas in canvases:

            # -----------------------------
            # 1. EXTRACT CANVAS METADATA
            # -----------------------------
            canvas_meta = {
                "arkiv": "",
                "serie": "",
                "referenskod": "",
                "datering": "",
                "anmärkning": "",
                "sok_url": None
            }

            for m in canvas.get("metadata", []):
                label = self._safe_label(m)
                value = self._safe_value(m)

                if label == "Bildid":
                    canvas_meta["image_id"] = value
                elif label == "Länk":
                    canvas_meta["sok_url"] = value
                elif label == "Källhänvisning":
                    # This field contains ALL metadata in one string
                    # Example:
                    # "Kommerskollegium, Fjärde serien, SE/RA/420132/3/D/D a/Da 4/15 (1866-1866)"
                    self._parse_kallhanvisning(canvas_meta, value)

            # -----------------------------
            # 2. FALLBACK: MANIFEST METADATA
            # -----------------------------
            for key in ["arkiv", "serie", "referenskod", "datering", "anmärkning"]:
                if not canvas_meta[key]:
                    canvas_meta[key] = manifest_meta.get(key, "")

            # -----------------------------
            # 3. FALLBACK: EAD METADATA
            # -----------------------------
            for key in ["referenskod", "arkiv", "serie", "datering"]:
                if not canvas_meta[key]:
                    canvas_meta[key] = ead_meta.get(key, "")

            # -----------------------------
            # 4. PROCESS ANNOTATIONS
            # -----------------------------
            for annotation_page in canvas.get("items", []):
                #folding(annotation_page)
                for annotation in annotation_page.get("items", []):
                    #folding(annotation)
                    if images_downloaded >= max_per_manifest:
                        return rows, images_downloaded

                    image_url = annotation.get("body", {}).get("id")
                    if not image_url:
                        continue

                    # Extract filename
                    match = re.search(r"!(.*)/full", image_url)
                    if not match:
                        continue

                    image_id = match.group(1)
                    filename = f"{image_id}.jpg"
                    file_path = os.path.join(folder, filename)

                    if not self._download_file(image_url, file_path):
                        continue

                    # -----------------------------
                    # 5. BUILD ROW (PER IMAGE)
                    # -----------------------------
                    row = {
                        "image_id": image_id,
                        "local_filename": filename,
                        "image_url": image_url,
                        "sok_url": canvas_meta["sok_url"],
                        "arkiv": canvas_meta["arkiv"],
                        "serie": canvas_meta["serie"],
                        "referenskod": canvas_meta["referenskod"],
                        "datering": canvas_meta["datering"],
                        "anmärkning": canvas_meta["anmärkning"],
                        "pid": pid,
                        "referenskod_ead": ead_meta.get("referenskod"),
                        "arkivenhetstyp": ead_meta.get("arkivenhetstyp"),
                        "titel": ead_meta.get("titel")
                    }

                    rows.append(row)
                    images_downloaded += 1

        return rows, images_downloaded

    def _safe_label(self, entry):
        if isinstance(entry.get("label"), dict):
            return next(iter(entry["label"].values()))[0]
        if isinstance(entry.get("label"), str):
            return entry["label"]
        return ""
    def _parse_kallhanvisning(self, meta, text):
        # Example:
        # "Kommerskollegium, Fjärde serien, SE/RA/420132/3/D/D a/Da 4/15 (1866-1866)"
        parts = text.split(",")
        if len(parts) >= 3:
            meta["arkiv"] = parts[0].strip()
            meta["serie"] = parts[1].strip()
            ref = parts[2].strip()
            # Extract referenskod and datering
            match = re.search(r"(SE/RA/[^()]+)\s*\(([^)]+)\)", ref)
            if match:
                meta["referenskod"] = match.group(1)
                meta["datering"] = match.group(2)


    def _walk_collection(
        self,
        collection_json,
        pid,
        folder,
        ead_meta,
        max_per_collection,
        max_per_manifest,
        log_path,
        rows,
        manifest_index,
        total_images_downloaded,
        processed_manifests
    ):
        

        processed_manifests = set()
        for item in collection_json.get("items", []):
            #print(f"collection: {len(collection_json.get('items', []))}")
            #print(total_images_downloaded)
            # Stoppa om vi nått max_per_collection
            manifest_id = item.get("id")

            if manifest_id in processed_manifests:
                # vi har redan tagit bilder från denna volym
                continue

            #processed_manifests.add(manifest_id)

            if total_images_downloaded[0] >= max_per_collection:
                return

            item_json = self._safe_get_json(item.get("id"), pid=pid, log_path=log_path)
            if item_json is None:
                continue

            item_type = item.get("type")

            # FALL 1: Collection → rekursion
            if item_type == "Collection":
                self._walk_collection(
                    item_json,
                    pid,
                    folder,
                    ead_meta,
                    max_per_collection,
                    max_per_manifest,
                    log_path,
                    rows,
                    manifest_index,
                    total_images_downloaded,
                    processed_manifests
                )

            # FALL 2: Manifest → processa
            else:
                manifest_index[0] += 1

                manifest_rows, images_from_manifest = self._process_manifest(
                    item_json,
                    pid,
                    manifest_index[0],
                    folder,
                    max_per_manifest,
                    ead_meta,
                )

                rows.extend(manifest_rows)
                total_images_downloaded[0] += images_from_manifest

                # Stoppa om vi nått max_per_collection
                #print(total_images_downloaded[0])
                if total_images_downloaded[0] >= max_per_collection:
                    print(f"Limit exceeded: {total_images_downloaded[0]} images")
                    return




    def run(self, pid, folder, max_per_manifest=3, max_per_collection=10, ead_meta=None):
        log_path = os.path.join(folder, "broken_manifests.csv")

        collection = self._safe_get_json(f"{self.URL_BASE}{pid}", pid=pid, log_path=log_path)
        if collection is None:
            print(f"⚠️ Hoppar över PID {pid} pga trasig collection")
            return []

        os.makedirs(folder, exist_ok=True)

        rows = []
        manifest_index = [0]
        total_images_downloaded = [0]
        processed_manifests = set()

        self._walk_collection(
            collection,
            pid,
            folder,
            ead_meta,
            max_per_collection,
            max_per_manifest,
            log_path,
            rows,
            manifest_index,
            total_images_downloaded,
            processed_manifests
        )

        return rows




    def _safe_value(self, entry):
        """Return first available metadata value regardless of language key."""
        if "value" not in entry:
            return None

        v = entry["value"]

        if isinstance(v, dict):
            # return first list element from any key
            for key, val in v.items():
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
            return None

        # fallback
        return v



    def _log_broken_manifest(self, url, pid, error, log_path):
        """Append a row to a CSV log when a manifest cannot be parsed."""
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pid": pid,
            "manifest_url": url,
            "error": str(error),
        }

        file_exists = os.path.exists(log_path)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)



# ============================================================
# EADClient – OAI-PMH → EAD → metadata
# ============================================================

import requests
import urllib.parse
import xml.etree.ElementTree as ET


class EADClient:
    OAI_URL = "https://oai-pmh.riksarkivet.se/OAI"
    NS = "{urn:isbn:1-931666-22-9}"
    XLINK = "{http://www.w3.org/1999/xlink}"

    def fetch_ead(self, ref_code: str):
        ref = urllib.parse.quote_plus(ref_code)
        url = f"{self.OAI_URL}?verb=GetRecord&identifier={ref}&metadataPrefix=oai_ape_ead"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        tree = ET.fromstring(r.content)
        ead = tree.find(f".//{self.NS}ead")
        return ead

    # ---------------- Hjälpfunktioner (översättning av dina) ----------------

    def _base_data(self, ead, did, unit_type, out):
        unitid = did.find(f"./{self.NS}unitid")
        if unitid is not None and unitid.text:
            out["referenskod"] = unitid.text

        if unit_type is not None:
            out["arkivenhetstyp"] = unit_type.capitalize()

        unittitle = did.find(f"./{self.NS}unittitle")
        if unittitle is not None and unittitle.text:
            out["titel"] = unittitle.text

        unitdate = did.find(f"./{self.NS}unitdate")
        if unitdate is not None and unitdate.text:
            out["datering"] = unitdate.text

        url_ref = ead.find(
            f"./{self.NS}archdesc//{self.NS}otherfindaid/"
            f"{self.NS}p/{self.NS}extref"
        )
        if url_ref is not None:
            url = url_ref.attrib.get(f"{self.XLINK}href")
            if url:
                out["persistent_id"] = url.split("/")[-1]
                out["url"] = url

        for link in ead.findall(
            f"./{self.NS}archdesc/{self.NS}did/{self.NS}dao"
        ):
            role = link.attrib.get(f"{self.XLINK}role")
            href = link.attrib.get(f"{self.XLINK}href")
            if role == "IMAGE":
                out["bildvisning"] = href
            elif role == "MANIFEST":
                out["iiif_manifest"] = href

    def _institution_data(self, header, out):
        if header is None:
            return
        eadid = header.find(f"./{self.NS}eadid")
        if eadid is not None:
            institution = eadid.attrib.get("mainagencycode")
            if institution:
                institution = institution.replace("-", "/")
                out["arkivinstitution"] = {
                    "referenskod": institution,
                    "url": (
                        "https://sok.riksarkivet.se/nad"
                        f"?postid=ArkisRef+{urllib.parse.quote_plus(institution)}"
                        "&type=2&s=balder"
                    ),
                }

    def _origin_data(self, did, out):
        origin = did.find(
            f"./{self.NS}origination/{self.NS}corpname"
        )
        if origin is None:
            origin = did.find(
                f"./{self.NS}origination/{self.NS}persname"
            )
        if origin is not None:
            origin_data = {}
            if origin.text:
                origin_data["namn"] = origin.text
            origin_refcode = origin.attrib.get("authfilenumber")
            if origin_refcode:
                origin_data["referenskod"] = origin_refcode
                origin_data["url"] = (
                    "https://sok.riksarkivet.se/"
                    f"?postid=ArkisRef+{urllib.parse.quote_plus(origin_refcode)}"
                )
            out["arkivbildare"] = origin_data

    def _extent_data(self, did, out):
        extent = did.find(
            f"./{self.NS}physdesc/{self.NS}extent"
        )
        if extent is not None and extent.text:
            unit = extent.attrib.get("unit", "")
            out["omfång"] = f"{extent.text} {unit}".strip()

    def _restrict(self, ref, key, out):
        if ref is None:
            return

        data = {}
        specification = ref.attrib.get("type")
        if specification:
            data["omfattning"] = specification

        note = ref.find(f"./{self.NS}p")
        if note is not None:
            if note.text:
                data["anmärkning"] = note.text
            else:
                extref = note.find(f"./{self.NS}extref")
                if extref is not None:
                    href = extref.attrib.get(f"{self.XLINK}href")
                    if href:
                        data["anmärkning"] = href

        all_restrict = out.get("villkor")
        if all_restrict is None:
            all_restrict = {}
            out["villkor"] = all_restrict
        all_restrict[key] = data

    def _unit_type(self, unit_root):
        t = unit_root.attrib.get("otherlevel")
        if t is None:
            t = unit_root.attrib.get("level")
            if t is not None:
                t = {"series": "serie"}.get(t)
        if t is None:
            ref_code_el = unit_root.find(
                f"./{self.NS}did/{self.NS}unitid"
            )
            ref_code = ref_code_el.text if ref_code_el is not None else ""
            part_count = len(ref_code.split("/"))
            if part_count == 3:
                t = "arkiv"
            elif part_count == 4:
                t = "serie"
            else:
                t = "detaljnivå: volym, karta/ritning etc."
        return t

    # ---------------- Huvudfunktion (motsvarar unit_data) ----------------

    def parse_unit(self, ead, unit_root, include_children=True):
        NS = self.NS
        archdesc = ead.find(f"./{NS}archdesc")
        out = {}

        did = unit_root.find(f"./{NS}did")
        if did is not None:
            self._base_data(ead, did, self._unit_type(unit_root), out)

            header = ead.find(f"./{NS}eadheader")
            self._institution_data(header, out)
            self._origin_data(did, out)
            self._extent_data(did, out)

        access = archdesc.find(f"./{NS}accessrestrict") if archdesc is not None else None
        self._restrict(access, "åtkomst", out)

        use = archdesc.find(f"./{NS}userestrict") if archdesc is not None else None
        self._restrict(use, "användning", out)

        if include_children:
            children = []

            for child in unit_root.findall(f"./{NS}dsc/{NS}c"):
                children.append(self.parse_unit(ead, child, True))

            for child in unit_root.findall(f"./{NS}c"):
                children.append(self.parse_unit(ead, child, True))

            if children:
                out["innehåll"] = children

        return out

    def parse_units(self,ref_code:str,include_children:bool=True):
        ead = self.fetch_ead(ref_code=ref_code)
        archdesc = ead.find(f"{self.NS}archdesc")
        out = self.parse_unit(ead,archdesc,include_children)
        return out


# ============================================================
# ArchiveGetter – main (container)
# ============================================================

class ArchiveGetter:
    def __init__(self):
        self.records = RecordsAPI()
        self.iiif = IIIFClient()
        self.ead = EADClient()

    def get_images(self, search: str, path: str, max_pids: int = None,
                   max_per_collection:int = 50, max_per_manifest: int = 4,
                   max_total:int = 1000):
        """
        Hämtar bilder via IIIF baserat på sökresultat + EAD-villkor.
        path: output-folder (skapas om den inte finns)
        max_pids: max antal PID:er att hämta totalt
        max_per_manifest: max antal bilder per manifest (skickas vidare till IIIFClient)
        """
        os.makedirs(path, exist_ok=True)
        hits = self.records.get_items(text=search)
        #print(hits)
        print(f"Antal träffar: {len(hits)}")
        # Alltid randomisera ordningen
        random.shuffle(hits)

        # Begränsa antal PID:er om max_pids är satt
        if max_pids is not None and max_pids<len(hits):
            hits = random.sample(hits,max_pids)
        else:
            max_pids=len(hits)
 
        
        print(f"Output-folder: {path}")
        print(f"Max PID-hämtningar: {max_pids}")
        print(f"Max bilder per kollektion: {max_per_collection}")
        print(f"Max bilder per manifest: {max_per_manifest}")

        k = 0  

        # CSV-path
        folder_name = os.path.basename(os.path.normpath(path))
        csv_path = os.path.join(path, f"{folder_name}_images.csv")

        # om CSV finns → läs in
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        start_len = len(df)


        for l,item in enumerate(hits):
            if k >= max_pids:
                print("Max antal PID-hämtningar uppnått.")
                break

            if "image" not in item.get("_links", {}):

                label = (
                    item.get("caption")
                    or item.get("titel")
                    or item.get("title")
                    or item.get("name")
                    or item.get("pid")
                    or "okänd post")

                print(f"Inga bildlänkar för {label}")
                continue

            metadata = item.get("metadata", {})
            ref = metadata.get("referenceCode", "")

            if not ref:
                continue

            out = self.ead.parse_units(ref)

            children = out.get("innehåll")

            if not children:
                villkor = out.get("villkor", {})

                if self._should_download(villkor):
                    pid = out.get("persistent_id")
                    if pid:
                        print(f"Hämtar PID nr {l} av {max_pids}: {pid}")

                        ead_meta = {
                            "referenskod": out.get("referenskod"),
                            "arkivenhetstyp": out.get("arkivenhetstyp"),
                            "titel": out.get("titel"),
                        }

                        rows = self.iiif.run(
                            pid,
                            path,
                            max_per_collection=max_per_collection,
                            max_per_manifest=max_per_manifest,
                            ead_meta=ead_meta,
                        )
                        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                        k += 1
                        stopper_length = len(df)-start_len
                        if stopper_length >= max_total:
                            print(f"Maximal number of images reached: {stopper_length}")
                            break
                    else:
                        print("Ingen PID")


            else:
                for inn in children:
                    if k >= max_pids:
                        break

                    villkor = inn.get("villkor", {})

                    if self._should_download(villkor):
                        pid = inn.get("persistent_id")
                        if pid:
                            print(f"Hämtar PID: {pid}")

                            ead_meta = {
                                "referenskod": inn.get("referenskod"),
                                "arkivenhetstyp": inn.get("arkivenhetstyp"),
                                "titel": inn.get("titel"),
                            }

                            rows = self.iiif.run(
                                pid,
                                path,
                                max_per_collection=max_per_collection,
                                max_per_manifest=max_per_manifest,
                                ead_meta=ead_meta,
                            )
                            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                            stopper_length = len(df)-start_len
                            if stopper_length >= max_total:
                                print(f"Maximal number of images reached: {stopper_length}")
                                break
                            k += 1
                        else:
                            print("Ingen PID")



        df.to_csv(csv_path, index=False, encoding="utf-8")
        end_len = len(df)
        print(f"CSV uppdaterad: {csv_path} med {end_len-start_len} nya bilder")


        csv_image_rows = df[df["local_filename"].str.lower().str.endswith(".jpg")]
        num_csv_images = len(csv_image_rows)

        jpg_files = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]
        num_jpg = len(jpg_files)

        print(f"CSV har {num_csv_images} bildrader, mappen har {num_jpg} bildfiler.")

        missing_in_csv = sorted(set(jpg_files) - set(csv_image_rows["local_filename"]))
        missing_on_disk = sorted(set(csv_image_rows["local_filename"]) - set(jpg_files))

        if missing_in_csv:
            print("Bilder på disk som saknas i CSV:")
            for f in missing_in_csv:
                print("  -", f)

        if missing_on_disk:
            print("Rader i CSV som saknar bildfil:")
            for f in missing_on_disk:
                print("  -", f)

        if not missing_in_csv and not missing_on_disk:
            print("CSV och mapp matchar perfekt.")
        self.find_duplicate_image_rows(csv_path)
        #self.zip_folder(path)

                
    def get_info(self, search: str, max_pids: int = 1):#, max_per_manifest: int = 1):
        """
            Hämtar info om arkiv/volym/serie via IIIF baserat på sökresultat + EAD-villkor.
            path: output-folder (skapas om den inte finns)
            max_pids: max antal PID:er att hämta totalt
            max_per_manifest: max antal bilder per manifest (skickas vidare till IIIFClient)
        """
        #os.makedirs(path, exist_ok=True)
        hits = self.records.get_items(search=search)
        #if len(hits)>max_pids:
        #    hits = random.sample(hits, max_pids)
        #folding(hits)
        print(f"Antal träffar: {len(hits)}")
        #print(f"Output-folder: {path}")
        print(f"Max PID-hämtningar: {max_pids}")
        #print(f"Max bilder per manifest: {max_per_manifest}")

        k = 0  # räknar antal PID:er som hämtats

        print("ALL ITEMS: \n")
        #folding(hits)
        #folding(hits[0])
        for it in hits:
            label = (
            it.get("caption")
            or it.get("titel")
            or it.get("title")
            or it.get("name")
            or it.get("pid")
            or "okänd post")
            if label != "okänd post":
                print(label)

        for item in hits:
            if k >= max_pids:
                #print("Max antal PID-hämtningar uppnått.")
                break
            #folding(item)
            # endast poster med bildlänk
            if "image" not in item.get("_links", {}):
                #print(item.keys())
                label = (
                    item.get("caption")
                    or item.get("titel")
                    or item.get("title")
                    or item.get("name")
                    or item.get("pid")
                    or "okänd post")

                print(f"Inga bildlänkar för {label}")
                #continue

            metadata = item.get("metadata", {})
            ref = metadata.get("referenceCode", "")
            print("############ Metadata: ##############\n")
            folding(metadata)
            print("\n############# INNEHÅLL: #################\n")
            out = self.ead.parse_units(ref)
            folding(out)
            children = out.get("innehåll",[])
            k+=1
            #folding(children)

    def _should_download(self, villkor: dict) -> bool:
        """
        Returnerar True om villkor är tomt eller om något värde matchar
        våra godkända nedladdningsvillkor.
        """
        # 1. Tomma villkor → alltid OK
        if not villkor:
            return True

        # 2. Annars loopa igenom alla värden
        allowed = {"nej", "dao", "ö", "delvis",}

        for _, value in villkor.items():
            for _, v in value.items():
                if v is None:
                    #print(v)
                    return True  # None räknas som OK
                if isinstance(v, str) and v.lower() in allowed:
                    #print(v)
                    return True
                #else:
                #    print(v)

        return False


    def zip_folder(self,folder_path, zip_path=None):
        """Zip an entire folder into a .zip file."""
        import zipfile
        if zip_path is None:
            zip_path = folder_path.rstrip("/\\") + ".zip"

        print(f"Skapar ZIP: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, folder_path)

                    # undvik att zippa zip-filen själv
                    if rel_path.endswith(".zip"):
                        continue

                    zipf.write(full_path, rel_path)

        print("✔️ ZIP klar")
        return zip_path


    def find_duplicate_image_rows(self, csv_path):
        """
        Hittar och tar bort dubletter i CSV baserat på local_filename.
        Behåller första förekomsten.
        Returnerar en DataFrame med alla borttagna dublettrader.
        """

        df = pd.read_csv(csv_path)

        # Filtrera rader som representerar bilder
        df_images = df[df["local_filename"].astype(str).str.lower().str.endswith(".jpg")]

        # Hitta alla dubletter (alla förekomster, inte bara extra)
        duplicates = df_images[df_images.duplicated("local_filename", keep=False)]

        if duplicates.empty:
            print("Inga dubletter hittades.")
            return duplicates

        print(f"Antal dubletter totalt (alla förekomster): {len(duplicates)}")
        print("Följande filer förekommer mer än en gång:")

        for fname, group in duplicates.groupby("local_filename"):
            print(f"  - {fname} ({len(group)} gånger)")

        # Hitta de rader som ska tas bort (alla utom första)
        to_remove = df_images[df_images.duplicated("local_filename", keep="first")]

        print(f"\nTar bort {len(to_remove)} dublettrader...")

        # Ta bort dublettraderna från original-DF
        df_clean = df.drop(to_remove.index)

        # Spara tillbaka
        df_clean.to_csv(csv_path, index=False)
        print(f"Rensad CSV sparad: {csv_path}")

        return to_remove

    def find_missing_images(self,csv_path, image_folder, output_missing_csv=None):
        """
        Hittar rader i CSV där bildfilen inte finns i mappen.
        csv_path: sökväg till CSV
        image_folder: mapp med .jpg-filer
        output_missing_csv: om du vill spara resultatet
        """

        # Läs CSV
        df = pd.read_csv(csv_path)

        # Filtrera rader som faktiskt representerar bilder
        df_images = df[df["local_filename"].str.lower().str.endswith(".jpg")]

        # Lista filer på disk
        disk_files = set(f for f in os.listdir(image_folder) if f.lower().endswith(".jpg"))

        # Lista filer i CSV
        csv_files = set(df_images["local_filename"].astype(str).tolist())

        # Hitta rader där fil saknas
        missing_on_disk = csv_files - disk_files

        print(f"Antal bildrader i CSV: {len(df_images)}")
        print(f"Antal bildfiler i mappen: {len(disk_files)}")
        print(f"Antal saknade bilder: {len(missing_on_disk)}")

        if missing_on_disk:
            print("\nRader i CSV som saknar bildfil:")
            for f in sorted(missing_on_disk):
                print("  -", f)

            # Extrahera raderna
            missing_rows = df_images[df_images["local_filename"].isin(missing_on_disk)]

            if output_missing_csv:
                missing_rows.to_csv(output_missing_csv, index=False)
                print(f"\nSparade saknade rader till: {output_missing_csv}")

            return missing_rows

        else:
            print("\nAlla bildrader i CSV har motsvarande filer på disk.")
            return pd.DataFrame()

    def remove_images(self,folder_path, extension=".jpg"):
        """
        Tar bort alla filer med angiven filändelse i en given mapp.
        
        Parameters:
            folder_path (str eller Path): Sökvägen till mappen.
            extension (str): Filändelsen att ta bort, t.ex. ".jpg" eller ".png".
        """
        folder = Path(folder_path)
        extension = extension.lower()

        if not folder.exists():
            print(f"Mappen finns inte: {folder}")
            return

        count = 0
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() == extension:
                file.unlink()
                count += 1

        print(f"Raderade {count} filer med ändelsen {extension} i {folder}")
        return count