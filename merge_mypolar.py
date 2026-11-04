#!/usr/bin/env python3
"""
merge_mypolar.py - Merge the class's Assignment-2 myPolar annotations into
  Stage 1 : one master file with full metadata (annotator, source link, ...)
            -> stage1_myPolar_full.csv  (never overwrites stage1_myPolar_full_clean.csv)
Stage 2 is built separately by build_stage2.py from stage1_myPolar_full_clean.csv.

Usage
  python3 merge_mypolar.py --input "<folder with the 23 *.zip files>" --output myPolar_merged

Requirements: pandas, scikit-learn
Optional    : myanmartools + pyidaungsu  (Zawgyi detection + conversion to Unicode)
              pip install pandas scikit-learn myanmartools pyidaungsu

All group decisions are in the CONFIG section below - change them there and re-run.
"""
import argparse, csv, io, json, re, sys, unicodedata, zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================== CONFIG ==
SEED = 42
SPLIT = (0.70, 0.15, 0.15)            # train / dev / test

# POLAR rule: if polarization == 0, all type + manifestation labels must be 0.
# Several annotators used the type columns as a *topic tag* for non-polarized
# text. True = zero them out in Stage 2 (Stage 1 keeps the original labels).
ENFORCE_POLAR_RULE = True

# Polarized rows whose Task-3 labels are all 0 are most likely "not annotated"
# (e.g. HanZawHtet web: 184/198). True = leave them out of the Task-3 file.
DROP_POLARIZED_WITHOUT_T3 = True
DROP_POLARIZED_WITHOUT_T2 = True

# Submissions kept in Stage 1 but excluded from training (reason is recorded).
EXCLUDE_FROM_STAGE2 = {
    "AungKoKoOo":    "no POLAR annotation: all labels 0, automated toxicity labels instead",
    "HanZawHtet_yt": "unlabelled (id, source, text only)",
}

# zip file -> list of (member path inside zip, annotator, part, reader)
# reader: "std" = 16-col class template, or a special reader name below.
SOURCES = {
    "AUNG_KO_KO_OO _Assignment2_Final.zip": [("Task3_Fully_Labelled_POLAR_Data-ID_Update.csv", "AungKoKoOo", "", "akko")],
    "AintKyiPhyuShin_Assignment2.zip": [("AintKyiPhyuShin_Assignment2/annotations.csv", "AintKyiPhyuShin", "", "std")],
    "AungChanNyein.Assignment-2.zip": [("AungChanNyein.Assignment-2/AungChanNyein_Assignment2.csv", "AungChanNyein", "", "acn")],
    "HanZawHtet-Assignment2.zip": [
        ("HanZawHtet-Assignment2/HanZawHtet-web-online.csv", "HanZawHtet", "web", "std"),
        ("HanZawHtet-Assignment2/HanZawHtet-youtube-comment.csv", "HanZawHtet", "yt", "std")],
    "HtooEaindraTin_Assignment2.zip": [("HtooEaindraTin_Assignment2.csv", "HtooEaindraTin", "", "std")],
    "KhineZarThwe_Assignment2.zip": [("KhineZarThwe_Assignment2/final_600.csv", "KhineZarThwe", "", "std")],
    "KyawZayaNaing_Assignment2.zip": [("polar_submission.csv", "KyawZayaNaing", "", "noheader9")],
    "MTS_Assignment-2.zip": [("MTS_Assignment-2/Final_Task_2.csv", "MTS", "", "mts")],
    "MyoThet_Assignment_2.zip": [
        ("MyoThet_Assignment_2/facebook/annotations.csv", "MyoThet", "facebook", "std"),
        ("MyoThet_Assignment_2/myanmar_rap_song/annotations.csv", "MyoThet", "rap_song", "std"),
        ("MyoThet_Assignment_2/social_media/annotations.csv", "MyoThet", "social_media", "std")],
    "NawSayRiHtoo_Assignment2.zip": [("NawSayRiHtoo_Assignment2/annotations_nawsayrihtoo_online.csv", "NawSayRiHtoo", "", "std")],
    "OhnmarNyunt_Annotations_Assignment2.zip": [
        ("OhnmarNyunt_Political_annotations.csv", "OhnmarNyunt", "political", "std"),
        ("OhnmarNyunt_Religious_annotations.csv", "OhnmarNyunt", "religious", "std")],
    "PhyoHtetPwint_Assignment2.zip": [("PhyoHtetPwint_Assignment2.csv", "PhyoHtetPwint", "", "std")],
    "PhyoMyatOo_assignment2.zip": [("PhyoMyatOo.csv", "PhyoMyatOo", "", "std")],
    "PunKhattgar_PolarAnnotations_Assignment2.zip": [("PunKhattgar_PolarAnnotations_Assignment2/PunKhattgar_PolarAnnotations_Assignment2.csv", "PunKhattgar", "", "std")],
    "Su_Ye_Hlaing_Assignment_2.zip": [("Su_Ye_Hlaing_mya_polar_data_extension/mya_suyehlaing_combined_polar_600_final.csv", "SuYeHlaing", "", "std")],
    "Thida_Aye_Assignment2.zip": [("Thida_Aye_Assignment2/Thida_Aye_Assignment2.csv", "ThidaAye", "", "std")],
    "Wai_Yan_Htet_Aung_Assignment2.zip": [("Wai_Yan_Htet_Aung.csv", "WaiYanHtetAung", "", "std")],
    "YeLinTun_Assignment2.zip": [("YeLinTun_Assignment2/YeLinTun_Assignment2-Polar-Data-Annotation.csv", "YeLinTun", "", "std")],
    "htunaungkyaw-assignment2.zip": [("htunaungkyaw.csv", "HtunAungKyaw", "", "std")],
    "linnthant_assignment2.zip": [("polar_submission.csv", "LinnThant", "", "noheader9")],
    "soe_thandar_tint.assignment-2.zip": [("soe_thandar_tint.assignment-2/soe_thandar_tint_annotated_data.csv", "SoeThandarTint", "", "std")],
    "thant_sin_tun_assignment2.zip": [("final_annotation.csv", "ThantSinTun", "", "std")],
    "thein_kyaw_lwin.assignment-2.zip": [("thein_kyaw_lwin.assignment-2/annotations_theinkyawlwin_600.csv", "TheinKyawLwin", "", "std")],
}
# ========================================================================

T1 = ["polarization"]
T2 = ["political", "racial_ethnic", "religious", "gender_sexual", "other"]
T3 = ["stereotype", "vilification", "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"]
LABELS = T1 + T2 + T3

RENAME = {
    "racial/ethnic": "racial_ethnic", "gender/sexual": "gender_sexual",
    "racial": "racial_ethnic", "gender": "gender_sexual",
    "ID": "id", "Text": "text", "Task 1": "polarization", "Political": "political",
    "Racial": "racial_ethnic", "Religious": "religious", "Gender": "gender_sexual",
    "Other": "other", "Source Link": "source", "Keyphrase": "key_phrase", "keyword": "key_phrase",
}

log_lines = []
def log(msg):
    print(msg); log_lines.append(msg)

# ---------------------------------------------------------------- readers
def read_csv_tolerant(raw: bytes, **kw) -> pd.DataFrame:
    """Read CSV; skip malformed rows (wrong field count) and report them."""
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]          # drop blank lines
    if kw.get("header", True):
        header, body = rows[0], rows[1:]
    else:
        header, body = kw["names"], rows
    good = [r for r in body if len(r) == len(header)]
    bad = len(body) - len(good)
    return pd.DataFrame(good, columns=header), bad

def read_member(raw, reader):
    if reader == "noheader9":       # KyawZayaNaing / LinnThant: no header, no Task-3
        names = ["source", "text", "polarization"] + T2 + ["key_phrase"]
        df, bad = read_csv_tolerant(raw, header=False, names=names)
    else:
        df, bad = read_csv_tolerant(raw)
    df = df.rename(columns=RENAME)
    if reader == "acn":
        df["source"] = df["source_detail"]
    if reader == "mts":
        df["platform_hint"] = df["source"]
        df["source"] = df["source_url"]
    return df, bad

# ------------------------------------------------------------- text tools
try:
    from myanmartools import ZawgyiDetector
    import pyidaungsu as pds
    _zg = ZawgyiDetector()
    def fix_encoding(s):
        if _zg.get_zawgyi_probability(s) > 0.95:
            return pds.cvt2uni(s), True
        return s, False
except Exception:
    log("WARNING: myanmartools/pyidaungsu not installed - Zawgyi text will NOT be converted.")
    def fix_encoding(s):
        return s, False

ZW = re.compile(r"[​‌‍⁠﻿]")
MM = r"က-႟ꩠ-ꩿꧠ-꧿"
SPACE_BETWEEN_MM = re.compile(rf"(?<=[{MM}])\s+(?=[{MM}])")

def clean_text(s):
    s = unicodedata.normalize("NFC", str(s))
    s = ZW.sub("", s)
    s = s.replace(",", " ").replace("\uff0c", " ")          # remove commas
    s = re.sub(r"\s+", " ", s).strip()
    return s

def unsegment(s):
    """Remove spaces between Burmese characters -> undo different word segmentations."""
    return SPACE_BETWEEN_MM.sub("", s)

# Syllable breaking (sylbreak, Ye Kyaw Thu)
_cons, _en = "က-အ", "a-zA-Z0-9"
_other = "ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\\[-`{-~\\s"
_SYL = re.compile(r"((?<!္)[" + _cons + r"](?![်္])|[" + _en + _other + r"])")
def syllable_segment(s):
    s = _SYL.sub(r" \1", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_key(s):
    """Key for duplicate detection: no spaces, no punctuation."""
    return re.sub(r"[\s၊။.,!?'\"“”‘’()\-–—:;…]", "", s)

def platform_of(url, part, hint=""):
    u = str(url).lower()
    for key, name in [("facebook", "facebook"), ("fb.", "facebook"), ("youtube", "youtube"),
                      ("youtu.be", "youtube"), ("tiktok", "tiktok"), ("wikipedia", "wikipedia"),
                      ("reddit", "reddit"), ("x.com", "twitter_x"), ("twitter", "twitter_x"),
                      ("huggingface", "huggingface_dataset"), ("telegram", "telegram")]:
        if key in u:
            return name
    if u.startswith("http"):
        return "website/news"
    if part in ("rap_song",):
        return "song_lyrics"
    if any(k in u for k in ("book", "စာအုပ်", "literature", "offline")):
        return "book/offline"
    return (hint or "unknown").lower() if hint else "unknown"

def domain_of(url):
    try:
        d = urlparse(str(url).split()[0]).netloc
        return d.replace("www.", "") if d else ""
    except Exception:
        return ""

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder containing the submission zip files")
    ap.add_argument("--output", default="myPolar_merged")
    a = ap.parse_args()
    inp, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. load every submission into one common schema
    frames = []
    for zname, members in SOURCES.items():
        zp = inp / zname
        if not zp.exists():
            log(f"MISSING zip: {zname}"); continue
        with zipfile.ZipFile(zp) as z:
            for member, annot, part, reader in members:
                df, bad = read_member(z.read(member), reader)
                key = f"{annot}_{part}" if part else annot
                df["annotator"], df["part"], df["source_file"] = annot, part, f"{zname}/{member}"
                df["_key"] = key
                for c in ["id", "source", "text", "key_phrase", "platform_hint"] + LABELS:
                    if c not in df:
                        df[c] = pd.NA
                frames.append(df[["annotator", "part", "_key", "source_file", "id", "source",
                                  "platform_hint", "text", "key_phrase"] + LABELS])
                log(f"loaded {key:<22} rows={len(df):>4}" + (f"  (skipped {bad} malformed rows)" if bad else ""))
    df = pd.concat(frames, ignore_index=True)

    # ---------- 2. clean text + labels
    df["text_original"] = df["text"].astype(str)
    fixed = df["text_original"].map(lambda s: fix_encoding(clean_text(s)))
    df["text"] = [clean_text(t) for t, _ in fixed]
    df["zawgyi_converted"] = [f for _, f in fixed]
    for c in LABELS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df[df["text"].str.len() > 0].copy()

    df["has_task1"] = df["polarization"].notna()
    df["has_task2"] = df[T2].notna().all(axis=1)
    df["has_task3"] = df[T3].notna().all(axis=1)

    pol1 = (df["polarization"] == 1).fillna(False).astype(bool)
    pol0 = (df["polarization"] == 0).fillna(False).astype(bool)
    t2sum = df[T2].fillna(0).sum(axis=1)
    t3sum = df[T3].fillna(0).sum(axis=1)
    checks = {
        "excluded_submission": df["_key"].isin(EXCLUDE_FROM_STAGE2),
        "no_labels": ~df["has_task1"],
        "no_task3_columns": ~df["has_task3"],
        "pol0_but_labels": pol0 & ((t2sum + t3sum) > 0),
        "pol1_no_type": pol1 & (t2sum == 0),
        "pol1_no_manifestation": pol1 & df["has_task3"] & (t3sum == 0),
    }
    df["issues"] = [";".join(k for k, m in checks.items() if m[i]) for i in df.index]

    # ---------- 3. ids, metadata, duplicates
    df["platform"] = [platform_of(s, p, h if isinstance(h, str) else "")
                      for s, p, h in zip(df["source"], df["part"], df["platform_hint"])]
    df["source_domain"] = df["source"].map(domain_of)
    df["text_nospace"] = df["text"].map(unsegment)
    df["dup_key"] = df["text_nospace"].map(norm_key)
    df["dup_count"] = df.groupby("dup_key")["dup_key"].transform("size")
    df["dup_annotators"] = df.groupby("dup_key")["annotator"].transform(lambda s: "|".join(sorted(set(s))))
    df = df.reset_index(drop=True)
    df["global_id"] = [f"mypolar_{i+1:05d}" for i in range(len(df))]

    stage1_cols = (["global_id", "annotator", "source_file", "id", "source",
                    "platform", "text_original", "text", "zawgyi_converted", "key_phrase"] + LABELS +
                   ["has_task3", "issues", "dup_count"])
    s1 = df[stage1_cols].rename(columns={"id": "original_id", "source": "source_url"})
    s1.to_csv(out / "stage1_myPolar_full.csv", index=False, encoding="utf-8-sig")
    log(f"\nStage 1 written: {len(s1)} rows -> stage1_myPolar_full.csv")

    log("Next: clean stage1_myPolar_full.csv by hand, save it as stage1_myPolar_full_clean.csv,")
    log("      then run: python3 build_stage2.py   (Stage 2 is built ONLY from the clean file)")
    (out / "stage1_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
