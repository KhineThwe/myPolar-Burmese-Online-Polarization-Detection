# myPolar: Burmese Online Polarization Detection

Polarization detection, polarization type classification and manifestation identification for Burmese, following the three subtasks of [POLAR @ SemEval-2026 Task 9](https://polar-semeval.github.io/).
Group project (Group 1, 12 members) for Assignment 7 of *AI Engineering (Fundamental), Batch 2*, taught by Dr. Ye Kyaw Thu. Presented on 11 October 2026.

## Tasks

| Task | Question | Labels |
|---|---|---|
| 1 | Is the sentence polarized? | binary |
| 2 | Who is the target? | political, racial/ethnic, religious, gender/sexual, other (multi-label) |
| 3 | How is it expressed? | stereotype, vilification, dehumanization, extreme language, lack of empathy, invalidation (multi-label) |

## Data

In Assignment 2, each student collected and annotated about 600 Burmese sentences (half from social media and online news, half from books, magazines and newspapers), with manual word segmentation and source links.

For this project the group merged them in two stages:

1. **Stage 1:** 23 submissions in 5 formats → one master file with text, labels, annotator, source link and platform, then manual cleaning.
2. **Stage 2:** model-ready train / dev / test files per task, without annotator or link columns.

| Step | Count |
|---|---|
| Sentences merged | 12,608 |
| Duplicates removed | 1,053 |
| Label fixes (non-polarized → type and manifestation = 0) | 1,972 |
| Zawgyi sentences converted to Unicode | 18 |
| Unlabelled sentences set aside | 586 |
| **Task 1 sentences after cleaning** | **10,955** (5,444 polarized, 5,511 not) |
| Task 2 / Task 3 sentences (polarized only) | 5,107 / 4,743 |

One shared 70 / 15 / 15 split is used for all tasks, so no test sentence appears in training for any task.

## Models

| Family | Models | Notebook |
|---|---|---|
| Traditional ML | SGD, linear SVM, Decision Tree, Random Forest; unigram / bigram × count / TF-IDF | `myPolar_ML.ipynb` |
| Deep learning from scratch | Bi-LSTM, 2-layer Transformer, syllable embeddings | `myPolar_DL.ipynb` |
| Pre-trained fine-tuning | XLM-RoBERTa base, mBERT | `mypolar-pretrained.ipynb` |

Model, features, best epoch and label thresholds are chosen on **dev**; each chosen model is scored on **test** once. Metric: macro-F1.

## Results (test macro-F1)

| Model | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| XLM-RoBERTa base | **0.802** | **0.677** | 0.479 |
| mBERT | 0.741 | 0.581 | **0.526** |
| Best traditional ML (SGD / linear SVM, bigram TF-IDF) | 0.784 | 0.662 | 0.512 |
| Bi-LSTM | 0.768 | 0.600 | 0.520 |
| Transformer from scratch | 0.696 | 0.531 | 0.518 |

All result tables are in [`results/`](results/).

## Key findings

1. **Data quality mattered most.** Annotators followed different rules (polarized rates from 0% to 99%), which had to be fixed before any model could learn.
2. **Burmese needs the right tokenizer.** Keeping syllables whole raised Task 1 dev macro-F1 from 0.63 to 0.80 with the same SVM, a bigger gain than any model change.
3. **Pre-training helps, but only a little.** XLM-RoBERTa beat a one-minute CPU linear model by about 0.02 on Tasks 1 and 2.
4. **Task 3 is the hardest.** Manifestation labels overlap; the best test macro-F1 is 0.526, and XLM-RoBERTa failed to learn it.

## Limitations and next steps
- Most sentences were labelled by one annotator, with different levels of strictness.
- Tasks 2 and 3 have only about 3,500 training sentences; *religious* has only 386 examples.
- Next: retrain XLM-RoBERTa on Task 3 with a lower learning rate and several seeds, agree on one annotation guideline, and label the 586 unlabelled sentences with model help.

## Team
Ohnmar Nyunt, Nandar Phyo, Aye Aye Aung, Phyu Pyar Chan Myae Kyaw, Khin Chaw Lai Lai Tun, Linn Thant, Khant Phone Naing, La Yaung Phyo, Moe Pyae Hein, Phyo Htet Pwint, Myat Min Ko Ko, Khine Zar Thwe.
Presenters: Khine Zar Thwe, Phyo Htet Pwint, Ohnmar Nyunt.

**My contribution (Khine Zar Thwe):** lead presenter, data merging script, cleaning, a model family, report.

## Repository layout

```
myPolar_ML.ipynb            traditional ML baselines
myPolar_DL.ipynb            Bi-LSTM and Transformer
mypolar-pretrained.ipynb    XLM-RoBERTa and mBERT fine-tuning (Kaggle GPU)
results/                    result tables (CSV)
slides/                     final presentation
data/                       not included (see data/README.md)
```
