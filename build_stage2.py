#!/usr/bin/env python3
"""
build_stage2.py - Build the model-ready Stage-2 files for POLAR Subtask 1 / 2 / 3
from the group's hand-cleaned Stage-1 file (stage1_myPolar_full_clean.csv).

Usage
  python3 build_stage2.py                                  # uses defaults below
  python3 build_stage2.py --input stage1_myPolar_full_clean.csv --output .

Requirements: pandas, scikit-learn

Output (in <output>/):
  stage2/task1_polarization{,_train,_dev,_test}.csv
  stage2/task2_type{,_train,_dev,_test}.csv
  stage2/task3_manifestation{,_train,_dev,_test}.csv
  stage2/unlabelled_for_inference.csv   (no labels - prediction demo only, never for scores)
  stage2_report.md, inter_annotator_agreement.csv, config_used.json

All group decisions are in the CONFIG section - change them there and re-run.
"""
import argparse, json, re, unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

# ============================================================== CONFIG ==
SEED = 42
SPLIT = (0.70, 0.15, 0.15)            # train / dev / test

# POLAR rule: if polarization == 0, all type + manifestation labels must be 0.
ENFORCE_POLAR_RULE = True
# Polarized rows with no Task-2 / Task-3 label at all are treated as "not annotated".
DROP_POLARIZED_WITHOUT_T2 = True
DROP_POLARIZED_WITHOUT_T3 = True

# Annotators whose labels are only trusted for some platforms.
# AungKoKoOo filled every POLAR label with 0 by default: his Wikipedia sentences are
# genuinely non-polarized and are kept; his social-media comments (simbolo hate-speech
# dataset) contain clearly polarized text, so they go to the UNLABELLED set instead.
TRUST_ONLY_PLATFORMS = {
    "AungKoKoOo": ["wikipedia"],
}
# Rows without a polarization label (e.g. HanZawHtet's YouTube comments) also go to the
# unlabelled set: stage2/unlabelled_for_inference.csv (prediction demo / manual check /
# self-training). They are never used for training or evaluation scores.
# ========================================================================

T1 = ["polarization"]
T2 = ["political", "racial_ethnic", "religious", "gender_sexual", "other"]
T3 = ["stereotype", "vilification", "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"]
LABELS = T1 + T2 + T3

# ------------------------------------------------------------- text tools
INVISIBLE = re.compile(r"[​-‏‪-‮⁠﻿­]")
MM = r"က-႟ꩠ-ꩿꧠ-꧿"
SPACE_BETWEEN_MM = re.compile(rf"(?<=[{MM}])\s+(?=[{MM}])")

EMOJI = re.compile("[\U0001F000-\U0001FAFF\U0001F1E6-\U0001F1FF\u2600-\u27BF\u2B00-\u2BFF"
                   "\u2300-\u23FF\uFE0F\uFE0E\u20E3\u200D]")
QUOTES = re.compile("[\"'\u2018\u2019\u201c\u201d`\u00b4]")
PARENS = re.compile(r"[()\uff08\uff09]")

def clean_text(s):
    s = unicodedata.normalize("NFC", str(s))
    s = INVISIBLE.sub("", s)
    s = EMOJI.sub(" ", s)                                    # emoji -> space
    s = QUOTES.sub("", s)                                    # ' " and curly quotes removed
    s = PARENS.sub(" ", s)                                   # ( ) -> space
    s = s.replace(",", " ").replace("\uff0c", " ")          # commas -> space
    return re.sub(r"\s+", " ", s).strip()

def unsegment(s):
    """Remove spaces between Burmese characters (undo different word segmentations)."""
    return SPACE_BETWEEN_MM.sub("", s)

# Syllable breaking (sylbreak, Ye Kyaw Thu)
_cons, _en = "က-အ", "a-zA-Z0-9"
_other = "ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\\[-`{-~\\s"
_SYL = re.compile(r"((?<!္)[" + _cons + r"](?![်္])|[" + _en + _other + r"])")
def syllable_segment(s):
    return re.sub(r"\s+", " ", _SYL.sub(r" \1", s)).strip()

def dup_key(s):
    return re.sub(r"[\s၊။.,!?'\"“”‘’()\-–—:;…]", "", s)

def to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="stage1_myPolar_full_clean.csv")
    ap.add_argument("--output", default=".")
    a = ap.parse_args()
    out = Path(a.output); (out / "stage2").mkdir(parents=True, exist_ok=True)
    log = []
    def say(m): print(m); log.append(m)

    # ---------- 1. read the cleaned Stage-1 file
    df = pd.read_csv(a.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = ["global_id", "annotator", "text"] + LABELS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {a.input}: {missing}")
    if df["global_id"].duplicated().any():
        raise SystemExit(f"Duplicate global_id values: {df.loc[df.global_id.duplicated(), 'global_id'].tolist()[:10]}")
    df = df.sort_values("global_id").reset_index(drop=True)
    say(f"Read {len(df)} rows from {a.input}")

    for c in LABELS:
        bad = ~df[c].str.strip().isin(["", "0", "1", "0.0", "1.0"])
        if bad.any():
            raise SystemExit(f"Column {c} has values other than 0/1/blank, e.g. rows "
                             f"{df.loc[bad, 'global_id'].tolist()[:5]}: {df.loc[bad, c].unique()[:5]}")
        df[c] = pd.to_numeric(df[c].str.strip().replace("", None), errors="coerce").astype("Int64")

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df["has_task1"] = df["polarization"].notna()
    df["has_task2"] = df[T2].notna().all(axis=1)
    df["has_task3"] = df[T3].notna().all(axis=1)
    df["text_nospace"] = df["text"].map(unsegment)
    df["dup_key"] = df["text_nospace"].map(dup_key)

    # ---------- 2. labelled vs unlabelled rows
    if "platform" not in df.columns:
        df["platform"] = ""
    untrusted = pd.Series(False, index=df.index)
    for ann, plats in TRUST_ONLY_PLATFORMS.items():
        m = (df["annotator"] == ann) & ~df["platform"].isin(plats)
        untrusted |= m
        say(f"{ann}: keeping {int(((df.annotator == ann) & ~m).sum())} rows from {plats}; "
            f"{int(m.sum())} other rows -> unlabelled set")
    no_label = ~df["has_task1"] & ~untrusted
    say(f"Rows without a polarization label: {int(no_label.sum())} -> unlabelled set")
    unl = df[untrusted | no_label].copy()
    s2 = df[~untrusted & df["has_task1"]].copy()
    say(f"Labelled rows: {len(s2)}  |  unlabelled rows: {len(unl)}")

    if ENFORCE_POLAR_RULE:
        p0 = s2["polarization"] == 0
        n = int((p0 & (s2[T2 + T3].fillna(0).sum(axis=1) > 0)).sum())
        for c in T2 + T3:
            s2.loc[p0 & s2[c].notna(), c] = 0
        say(f"POLAR rule: zeroed type/manifestation labels on {n} non-polarized rows")

    # ---------- 3. inter-annotator agreement (before dedup)
    pairs = {}
    for _, g in s2[s2.duplicated("dup_key", keep=False)].groupby("dup_key"):
        g = g.drop_duplicates("annotator").sort_values("annotator")
        r = g.to_dict("records")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                pairs.setdefault((r[i]["annotator"], r[j]["annotator"]), []).append(
                    (int(r[i]["polarization"]), int(r[j]["polarization"])))
    iaa = []
    for (a1, a2), lst in pairs.items():
        if len(lst) < 20: continue
        y1, y2 = zip(*lst)
        k = cohen_kappa_score(y1, y2) if len(set(y1 + y2)) > 1 else float("nan")
        iaa.append({"annotator_1": a1, "annotator_2": a2, "shared": len(lst),
                    "pol_agreement": round(sum(u == v for u, v in lst) / len(lst), 3),
                    "cohen_kappa": round(k, 3)})
    iaa = pd.DataFrame(iaa).sort_values("shared", ascending=False) if iaa else pd.DataFrame()
    if len(iaa): iaa.to_csv(out / "inter_annotator_agreement.csv", index=False)

    # ---------- 4. dedup (keep the copy with Task-3 labels, else the first)
    before = len(s2)
    s2 = (s2.assign(_r=(~s2["has_task3"]).astype(int))
             .sort_values(["dup_key", "_r", "global_id"])
             .drop_duplicates("dup_key").drop(columns="_r").sort_values("global_id"))
    say(f"Dedup: {before} -> {len(s2)} rows ({before - len(s2)} duplicate texts removed)")
    s2["text_syllable"] = s2["text_nospace"].map(syllable_segment)

    # ---------- 5. one master split shared by all tasks
    tr, rest = train_test_split(s2.index, test_size=SPLIT[1] + SPLIT[2], random_state=SEED,
                                stratify=s2["polarization"])
    dv, te = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]), random_state=SEED,
                              stratify=s2.loc[rest, "polarization"])
    s2["split"] = "train"; s2.loc[dv, "split"] = "dev"; s2.loc[te, "split"] = "test"

    def write(name, frame, labels):
        f = frame[["global_id", "text_nospace", "text_syllable", "split"] + labels] \
                .rename(columns={"global_id": "id", "text_nospace": "text"})
        for c in labels: f[c] = f[c].astype(int)
        f.to_csv(out / "stage2" / f"{name}.csv", index=False, encoding="utf-8-sig")
        for sp in ["train", "dev", "test"]:
            f[f.split == sp].drop(columns="split").to_csv(
                out / "stage2" / f"{name}_{sp}.csv", index=False, encoding="utf-8-sig")
        return f

    t1 = write("task1_polarization", s2, T1)
    pol = s2[s2["polarization"] == 1]
    t2 = pol[pol["has_task2"]]
    if DROP_POLARIZED_WITHOUT_T2: t2 = t2[t2[T2].sum(axis=1) > 0]
    t2 = write("task2_type", t2, T2)
    t3 = pol[pol["has_task3"]]
    if DROP_POLARIZED_WITHOUT_T3: t3 = t3[t3[T3].sum(axis=1) > 0]
    t3 = write("task3_manifestation", t3, T3)
    say(f"Task1 {len(t1)} | Task2 {len(t2)} | Task3 {len(t3)}  -> {out / 'stage2'}")

    # ---------- unlabelled set for inference (no overlap with labelled data, no duplicates)
    n0 = len(unl)
    unl = unl[~unl["dup_key"].isin(set(s2["dup_key"]))].drop_duplicates("dup_key")
    unl["text_syllable"] = unl["text_nospace"].map(syllable_segment)
    cols = ["global_id", "annotator", "platform"] + (["source_url"] if "source_url" in unl else []) + \
           ["text_nospace", "text_syllable"]
    unl[cols].rename(columns={"global_id": "id", "text_nospace": "text"}) \
        .to_csv(out / "stage2" / "unlabelled_for_inference.csv", index=False, encoding="utf-8-sig")
    say(f"Unlabelled set: {n0} -> {len(unl)} rows after removing duplicates / overlap with labelled data "
        f"-> stage2/unlabelled_for_inference.csv  {unl.groupby('annotator').size().to_dict()}")

    # ---------- 6. report
    g = df.groupby("annotator")
    per = pd.DataFrame({
        "rows": g.size(),
        "polarized": g["polarization"].apply(lambda s: int((s == 1).sum())),
        "pol_rate_%": g["polarization"].apply(lambda s: round(100 * (s == 1).mean(), 1) if s.notna().any() else None),
        "in_stage2": s2.groupby("annotator").size(),
        "in_unlabelled_set": unl.groupby("annotator").size(),
    }).fillna({"in_stage2": 0, "in_unlabelled_set": 0})
    per["in_unlabelled_set"] = per["in_unlabelled_set"].astype(int)
    per["in_stage2"] = per["in_stage2"].astype(int)
    R = ["# Stage 2 build report\n",
         f"Input: {a.input}  |  seed {SEED}  |  split {SPLIT}  |  ENFORCE_POLAR_RULE={ENFORCE_POLAR_RULE}\n",
         "## Log\n", "```", *log, "```\n",
         "## Per annotator\n", "```", per.to_string(), "```\n",
         "## Inter-annotator agreement (Task 1, same sentence labelled by two people)\n",
         "```", iaa.to_string(index=False) if len(iaa) else "none", "```\n", "## Stage 2 files\n"]
    for name, f, labs in [("task1_polarization", t1, T1), ("task2_type", t2, T2), ("task3_manifestation", t3, T3)]:
        R.append(f"### {name}: {len(f)} rows  " + str(f["split"].value_counts().to_dict()))
        dist = pd.DataFrame({sp: f[f.split == sp][labs].mean().round(3) for sp in ["train", "dev", "test"]})
        dist["count_all"] = f[labs].sum()
        R += ["```", dist.to_string(), "```\n"]
    (out / "stage2_report.md").write_text("\n".join(R), encoding="utf-8")
    json.dump({"input": a.input, "seed": SEED, "split": SPLIT, "enforce_polar_rule": ENFORCE_POLAR_RULE,
               "drop_polarized_without_t2": DROP_POLARIZED_WITHOUT_T2,
               "drop_polarized_without_t3": DROP_POLARIZED_WITHOUT_T3,
               "trust_only_platforms": TRUST_ONLY_PLATFORMS},
              open(out / "config_used.json", "w"), indent=2)

if __name__ == "__main__":
    main()
