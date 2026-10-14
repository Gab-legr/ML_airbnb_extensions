# Airbnb Price Prediction in Europe: From Local Baseline to Domain Generalization

### Project Objectives
This project is built around two main scientific goals:

* **Applying Core ML Fundamentals (Case Study: Paris):** analyse the Parisian Airbnb market by applying, comparing and interpreting the six core methods studied in the course (PCA, LDA, CART, Bootstrap, Bagging, Random Forests), alongside one supplementary method (UMAP).

* **Scaling to Europe (Domain Generalization):** after establishing a local baseline, we measure how much a model loses on cities it has never seen (Rome, Madrid, Vienna) because of domain shift. We test three remedies with leave-one-city-out cross-validation and an ablation study:
  * per-city standardisation of the features;
  * domain weighting;
  * a reranking of the predictions.

  The question: can we estimate the price tier of listings in a brand-new city without any of its labelled data, relying only on general dynamics?

### Key results
* **Paris:** random forest **56.1% accuracy / 0.551 F1-macro** on four balanced classes (chance: 25%), ahead of bagging (0.537), the pruned CART (0.516) and LDA (0.472).
* **Unseen city** (leave-one-city-out, mean F1-macro):
  * **0.474** for the final pipeline (per-city standardisation + domain weighting + reranking);
  * 0.444 for a naive model trained on Paris only;
  * 0.609 when the model is trained inside the city (upper bound).
  * Per-city standardisation brings most of the gain; domain weighting has no measurable effect.

### Repository Structure
The project is structured sequentially. Data engineering and modelling are separated for a smooth reading experience.

```
├── data/                               # Raw Inside Airbnb dumps and processed datasets
├── data_processing/                    # Engineering & feature extraction
│   ├── city_selection.ipynb            # (Optional) Script to find usable datasets
│   ├── data_processing_paris.ipynb     # Step-by-step cleaning
│   └── data_processing_cities.ipynb    # Same pipeline applied to Rome, Madrid, Vienna
├── 1_ML_Paris.ipynb                    # Applied ML methods on Paris
├── 2_generalisation.ipynb              # Domain generalisation project
├── figures/                            # Every figure of the notebooks (+ all_figures.pdf)
├── results/                            # Cached grid searches and result tables
├── docs/
│   └── EXTENSIONS.md                   # Possible extensions (ML, DL, RL, econometrics)
├── extensions/pilot_experiments.py     # Quick runs of three extensions
└── README.md
```

## Reading Guide & Evaluation Checklist

Here is exactly where to find the methods required by the grading rubric:

### 1. The Core Curriculum: `1_ML_Paris.ipynb`
This notebook focuses exclusively on the Paris dataset. Every method uses the same 24 features and the same stratified 80/20 split, so the scores are comparable:

* **Dimensionality Reduction & Analysis:**
  * **PCA** (Principal Component Analysis) for variance analysis.
  * **UMAP** *(Supplementary Method)* for non-linear visualisation of the listings.
* **Base Classification Models:**
  * **LDA** (Linear Discriminant Analysis) for linear separation of the price quartiles.
  * **CART** (Classification and Regression Trees), pruned with the 1-SE rule, to extract interpretable decision rules.
* **Ensemble Methods & Resampling:**
  * **Bootstrap** for uncertainty estimation (confidence intervals of the test scores).
  * **Bagging** to reduce tree variance (number of trees chosen on the out-of-bag error).
  * **Random Forests** as our final baseline model for Paris.

### 2. The Advanced Approach: `2_generalisation.ipynb`
This notebook looks for patterns that hold independently of the city seen in training. It uses the course's Random Forest as a baseline and builds a domain-generalisation pipeline around it:

* **Domain shift diagnostics:** the feature distributions per city, and a classifier that recognises the city from the features.
* **Per-city standardisation:** each continuous feature is z-scored with the mean and standard deviation of its own city (no labels needed).
* **Domain Weighting:** sample weights so the largest city (Paris) does not dominate.
* **Posterior reranking:** each listing gets its expected class ($E[Y|X]$); these scores are ranked and cut into four equal groups to match the known 25% per class (a label-shift correction).
* **LODO-CV evaluation:** Leave-One-Domain-Out cross-validation with an ablation, a naive Paris-only baseline and an in-domain upper bound, backed by **McNemar statistical tests**.

### 3. Data Engineering: `data_processing/`
*(These notebooks do not need to be re-run: the generated data is provided in `data/`.)*

They document our engineering choices:
* keyword extraction for the amenities;
* price quartiles computed separately in each city;
* our handling of spatial data.

We used LLMs to help code some of the functions that extract and automate the feature engineering, to save time on the technical coding.

---

## Geospatial Feature Engineering

A major challenge in this project was the nature of the raw data. The classic algorithms of the course (PCA, LDA, CART) struggle to extract meaningful patterns from raw GPS coordinates (latitude/longitude). With raw coordinates, PCA would just recreate a map of Paris, and CART would memorise specific neighbourhoods instead of learning real-estate rules.

To apply the course's core methods, we abstracted the geospatial data. Raw GPS coordinates are converted into continuous features such as the **normalised distance to the city centre** (haversine distance in km divided by the city's 95th percentile). This translates the map into a language that standard statistical learning algorithms can process. It prevents massive geographic overfitting and makes the European domain generalisation possible.

LLMs and coding agents were used to debug code, to speed up some plotting/printing functions, to translate from French to English and to write some of the comments.

## How to run
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 1_ML_Paris.ipynb        # ~1 h
jupyter nbconvert --to notebook --execute --inplace 2_generalisation.ipynb  # ~30–40 min
```
The grid searches are cached in `results/`. Delete the CSV files there to recompute them.

## Authors

Group project for the TSE course "High-Dimensional Data Analysis and Machine Learning", by Gabriel ([Gab-legr](https://github.com/Gab-legr)), Felix ([Felix-prjcts](https://github.com/Felix-prjcts)), Elmi96 and nilsrastoul-eng. The original team repository, with the full commit history, is [Felix-prjcts/ML_Airbnb_domain_generalisation](https://github.com/Felix-prjcts/ML_Airbnb_domain_generalisation). This repository is my personal copy, where I continue the work.
