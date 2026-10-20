# Same Glyphs, Different Bytes — code and aggregate results (anonymized review copy)

This repository accompanies an ICEEICT 2027 submission. It contains the
perturbation generators, the normalization defenses with their unit tests,
the rendering-based visual check, the paired evaluation runner, the
statistical analysis, the training script for the XLM-R / MuRIL / BanglaBERT
classifiers, the Kaggle notebook, and the aggregate result tables. Raw data
and per-comment predictions are not included: BD-SHS is available from its
authors (LREC 2022) and BLP-2025 Task 1 from its organizers (CC BY-NC-SA 4.0).

## Layout
- `src/normalize/bn_norm.py` — defenses D0–D3 (rule 0: Bangla canonical composition; rules 1–7) + tests
- `src/perturb/families.py` — families F1–F5 and per-character F2 variants + tests
- `src/render/` — HarfBuzz/FreeType rendering, pixel comparison, visual gate
- `src/eval/run_eval.py` — paired clean-vs-perturbed evaluation with a text cache
- `src/stats/analyze.py` — outcomes, clustered bootstrap, Holm correction, LaTeX tables
- `src/train/finetune_xlmr.py` — fine-tuning (any sequence-classification backbone)
- `notebooks/` — the Kaggle notebook used for GPU runs
- `results/` — aggregate tables and summaries only

## Reproduce (CPU is enough for everything except fine-tuning)
```
python -m pytest src/normalize src/perturb -q
python src/render/visual_validation.py 60
python src/eval/run_eval.py --split test --out results/final --models <model_id_or_path>:1
python src/stats/analyze.py --dir results/final --split test
```
Fonts: Noto Sans Bengali (OFL), SolaimanLipi (ekushey.org), Kalpurush; place them in `fonts/`.
