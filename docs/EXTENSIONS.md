# Extensions — where the project can go next

This list is ordered by expected value for this project (the domain gap and the missing information in the features), not by how fashionable the method is. Each item says **why** it matters here, **how** to do it, and roughly how much work it is. Three of them were tried quickly; their results are in [Pilot results](#pilot-results) and come from `extensions/pilot_experiments.py`.

## Priority roadmap

| # | Extension | Family | Effort | Why it is worth it |
|---|---|---|---|---|
| 1 | Use the columns we dropped (review scores, host, availability) and the full amenity list | Features | 1 day | Cheapest source of new signal |
| 2 | Surface area and other attributes extracted from the description by an LLM | Features / NLP | 2–3 days | Surface area is the biggest missing price driver |
| 3 | Gradient boosting + ordinal target / log-price regression | ML | 1 day | Standard strong baseline; pilot gain small (+0.7 pt) |
| 4 | Leave-one-city-out hyperparameter selection | DG | ½ day | Current tuning optimises the wrong objective |
| 5 | Few-shot adaptation: a few labelled listings of the new city | DG | 1 day | Pilot: 200 local labels ≈ 125k listings from other cities |
| 6 | SHAP explanations compared across cities | Interpretability | 1 day | Turns "the drivers differ by city" into a figure |
| 7 | Label-shift estimation when the target price distribution is unknown | DG | 2–3 days | Removes the strongest assumption of the reranker |
| 8 | Domain-adversarial network (DANN) / invariant learning | Deep learning | 1 week | The textbook DG approach, a good comparison |
| 9 | Photos and description embeddings (multimodal) | Deep learning | 1–2 weeks | Tests the "middle classes differ by visual cues" hypothesis |
| 10 | Dynamic pricing with contextual bandits / RL in a simulator | RL | 2+ weeks | Moves from *predicting* a price class to *setting* a price |
| 11 | Olympics / regulation effect with difference-in-differences | Econometrics | 1 week | A causal question the data can answer |
| 12 | Package, tests, pipeline, demo app | Engineering | 2–3 days | Reproducibility; tests would have caught the bugs found in review |

---

## 1. Better models on the current features (machine learning)

**1.1 Gradient boosting.** LightGBM, XGBoost, CatBoost or scikit-learn's `HistGradientBoostingClassifier` are usually stronger than random forests on tabular data. CatBoost also handles high-cardinality categorical features natively (the neighbourhood name, the exact `property_type`) without the frequency encoding used today. *Pilot: +0.7 point on Paris, nothing across cities (below).*

**1.2 Respect the order of the classes.** Budget < Standard < Premium < Luxury, but the forest treats the four classes as unordered labels. Options:
* regress `log(price)` and cut the prediction into quartiles. It uses the full price information instead of four bins, but the naive version lost 0.9 point in the pilot (below), so it needs tuned thresholds or an ordinal loss;
* ordinal models: proportional-odds (ordered logit) regression, or the CORAL/CORN losses for neural networks;
* report ordinal metrics as well: mean absolute error on the class index, quadratic weighted kappa.

**1.3 Tune for the real objective.** In notebook 2, the grid search uses 3-fold CV that mixes the source cities, so it picks the best configuration *in-distribution*. Scoring each configuration with `LeaveOneGroupOut` (groups = cities) selects for transfer. It typically favours more regularised forests (larger leaves, shallower trees).

**1.4 Spatial cross-validation inside a city.** Neighbouring listings are similar, so a random split is optimistic. Block CV by neighbourhood or by spatial tiles gives a more honest in-city score.

**1.5 Uncertainty.** Conformal prediction (e.g. `MAPIE`) returns a *set* of classes with guaranteed coverage ("Standard or Premium") instead of a single class. That is useful exactly where the model is weakest. Reliability diagrams would also show whether the predicted probabilities are calibrated.

**1.6 Interpretability.** SHAP values (TreeExplainer) per city would show *how* the drivers differ (e.g. the distance-to-centre effect in Rome versus Paris). Accumulated local effects (ALE) plots are safer than partial-dependence plots with correlated features such as `accommodates`, `beds` and `bedrooms`.

## 2. Richer features (largest expected gain)

The in-domain ceiling is about 0.55–0.68 F1-macro. The main reason is information the model never sees, not the algorithm.

**2.1 Columns already in the raw files but dropped:** `review_scores_*` (quality), host features (`host_is_superhost`, `calculated_host_listings_count` as a "professional host" signal), `availability_30/365`, `instant_bookable`, `license`, `reviews_per_month`. Add them one family at a time and check the cross-city score, since some are domain-specific (regulation differs by city).

**2.2 The full amenity list.** Each Paris listing has about 27 amenities on average; we keep six flags and six hand-made scores. A multi-hot encoding of the ~200 most frequent amenities (or a TF-IDF over amenity tokens) is simple and lets the model choose. Unit tests on the parsing would have caught the "dishwasher counted as washing machine" bug.

**2.3 Geography beyond the distance to the centre.**
* Distances to points of interest (metro stations, monuments, the river) from OpenStreetMap.
* H3 hexagons, or the average price of the k nearest listings (a "spatial lag"). The spatial lag must be computed out-of-fold so the listing's own price does not leak into its features.
* Distances in walking time rather than straight lines.

**2.4 Text.** Names and descriptions are written in French, Italian, Spanish, German or English. Multilingual sentence embeddings (e.g. `paraphrase-multilingual-MiniLM`, E5) map them to one space, which suits the cross-city setting.

**2.5 LLM-based extraction.** Ask an LLM to return structured fields from the description: **surface area (m²)**, floor, view, terrace, renovation, "luxury" wording. Surface area is the single strongest price driver in real estate and is missing from the data. Validate the extraction on a hand-labelled sample.

**2.6 Photos.** CLIP or ViT embeddings of the listing pictures. The notebooks explain the Standard/Premium confusion with "decor, light, overall feel"; photos are the way to test that hypothesis.

**2.7 Time.** Inside Airbnb publishes a scrape every quarter. Several scrapes per city give seasonality and remove the date confound of the current data (Paris June 2024 vs the others September 2025).

## 3. Domain generalisation and adaptation (the core topic)

**3.1 Few-shot adaptation.** How many labelled listings of a new city are needed to close the gap with the in-domain model? Train on the source cities plus *n* labelled target listings and plot the score against *n*. This answers a business question ("how much local data must we collect before launching in a new city?"). *Pilot: ~200 local labels already match the cross-city model (below).*

**3.2 Covariate-shift weighting.** Weight source listings by p(target | x) / p(source | x), estimated with a domain classifier (the one in notebook 2 already exists), or with KMM/KLIEP. This upweights source listings that look like the target city.

**3.3 Label-shift estimation.** The reranker assumes the target class shares are known (25%). For a realistic use case — predicting *absolute* price bands in a new city — that prior is unknown. EM re-estimation of the prior (Saerens et al., 2002) or Black-Box Shift Estimation (Lipton et al., 2018) estimate it from unlabelled target data.

**3.4 Invariant learning.** IRM, Group DRO, V-REx: train so that the model is good on *every* source city at once, not on average. The DomainBed benchmark gives reference implementations. A simpler version: keep only the features whose effect is stable across cities.

**3.5 Hierarchical models.** A Bayesian ordinal regression with city-level random effects (PyMC or Bambi) shares information across cities (partial pooling) and gives uncertainty on the city effects. It is also very interpretable.

**3.6 More cities.** Inside Airbnb covers 100+ cities. With 20–30 source cities one can:
* plot the out-of-domain score against the number of source cities;
* add city descriptors (GDP per capita, tourist arrivals, regulation) as features;
* try meta-learning.

**3.7 Test-time adaptation.** The per-city standardisation is a simple form of it (like AdaBN in deep learning). Matching quantiles per city (rank or quantile transform) instead of mean/std is a natural next step.

## 4. Deep learning

**4.1 Tabular deep learning.** MLP with embeddings, FT-Transformer, TabNet, SAINT; or **TabPFN**, a pretrained transformer for small tabular datasets that suits small cities such as Vienna. Expect gradient boosting to remain a strong baseline on this kind of data.

**4.2 Domain-adversarial training (DANN).** A shared encoder feeds two heads: a price-class head, and a city classifier connected through a gradient-reversal layer. The encoder learns features that predict price but not the city. Variants: Deep CORAL and MMD penalties. This is the deep-learning counterpart of the per-city standardisation and makes a strong interview topic.

**4.3 Multimodal model.** Combine tabular features, a text encoder (description) and an image encoder (photos), merged before the final layers (late fusion). Frozen pretrained encoders plus a small trained head are enough to start.

**4.4 Ordinal losses.** CORAL / CORN (rank-consistent ordinal regression) or soft-label / earth-mover's-distance losses. They penalise "Budget predicted as Luxury" more than "Budget predicted as Standard".

**4.5 Graph neural networks.** Build a k-nearest-neighbour graph of listings by location; GraphSAGE/GAT then propagates information from neighbours (price spillovers, local amenities).

**4.6 LLMs as predictors.** Give the listing card (structured fields + description) to an LLM, zero-shot or few-shot, and ask for the price class. This is an interesting cross-city baseline, since the LLM has general knowledge of each city. Watch the cost and compare honestly with the forest.

## 5. Reinforcement learning and pricing decisions

RL is a natural next step only if the question changes from *"which price class is this listing in?"* to *"what price should the host set?"*. RL needs actions (prices) and outcomes (bookings). Inside Airbnb only provides proxies: `calendar.csv` availability for occupancy, and reviews for bookings. A demand model or simulator is therefore needed.

**5.1 Contextual bandit for price recommendation.**
* Context = listing features + date; arms = a grid of nightly prices; reward = price × 1{booked}.
* Thompson sampling or LinUCB balance exploring prices against exploiting the best known one.
* The classifier from this project provides a prior: the price band where exploration should start.

**5.2 Full MDP in a simulator.** State = days before check-in, the calendar already booked, season, competitors' prices; action = tonight's price; reward = revenue. Algorithms: fitted Q-iteration or PPO. A multi-agent version (hosts competing in the same neighbourhood) is a research-level extension.

**5.3 Offline evaluation.** If logged pricing data were available, inverse propensity scoring or doubly-robust estimators evaluate a new pricing policy without deploying it. Confounding is the main risk: hosts set high prices on dates they already expect to be in demand.

**5.4 Active learning.** Choose which listings of a new city to label first (uncertainty or diversity sampling, possibly a learned acquisition policy). This combines with 3.1.

## 6. Econometrics and causal inference

**6.1 Hedonic price regression.** `log(price)` on characteristics with neighbourhood fixed effects. It gives interpretable coefficients (the price premium of air-conditioning, of an extra bedroom) to compare with the forest.

**6.2 Double machine learning.** Estimate the causal effect of one feature (e.g. professional host, self check-in, air-conditioning) on price while flexibly controlling for the others (`DoubleML`, `EconML`).

**6.3 Olympics effect.** Collect several Paris scrapes (before, during and after summer 2024) and use control cities to estimate the Olympics effect on prices and supply by difference-in-differences. The current data hints at it: 34% of Paris listings have no review, against 13–16% elsewhere.

**6.4 Regulation.** Paris caps entire-home rentals at 120 nights a year, and Madrid and Rome have their own rules. Difference-in-differences or synthetic control can estimate their effect on supply and prices.

## 7. Engineering and product

**7.1 Code structure.**
* Move the feature engineering into a `src/` package with unit tests, and share it between the Paris and city notebooks: today the two copies are maintained separately.
* Wrap the per-city standardiser in a scikit-learn transformer and use a `Pipeline`, so preprocessing cannot leak between train and test.

**7.2 Reproducibility.**
* Pin package versions: the committed virtual environment uses Python 3.14, while the notebooks were run with Python 3.12/3.13.
* Add one command that runs every notebook end-to-end (e.g. `nbclient` or `papermill`) and fixes the seeds.

**7.3 Data and experiments.** Keep the ~83 MB of CSV files out of git (DVC or git-lfs). Track experiments with MLflow.

**7.4 Demo.** A Streamlit or Gradio app where a user enters a listing and gets its price class, a SHAP explanation and comparable listings.

**7.5 Monitoring.** Re-score every new quarterly scrape. A domain classifier or the population stability index detects drift and triggers retraining.

---

## Pilot results

These quick runs use the corrected datasets. They are not tuned, so read them as signals, not final numbers. They come from `python extensions/pilot_experiments.py` (outputs in `results/pilot_*.csv` and `figures/extensions_pilot/`).

**1. Model family on Paris** (same 80/20 split as notebook 1, F1-macro on the test set; `results/pilot_paris_models.csv`)

| Model | F1-macro |
|---|---|
| Random forest (notebook 2 settings) | 0.550 |
| HistGradientBoosting, 600 iterations, early stopping | **0.557** |
| HistGradientBoosting regressor on log(price), cut at the training quartiles | 0.541 |

Gradient boosting gains less than one point without any tuning: on these features, the choice of algorithm is no longer the bottleneck. The naive log-price regression is slightly *worse*. Cutting a point prediction at fixed thresholds wastes the middle classes, so this route needs an ordinal loss or tuned thresholds before it can pay off.

**2. Gradient boosting across cities** (LODO, per-city standardisation + domain weights; `results/pilot_lodo_gradient_boosting.csv`)

| Model | Mean F1 raw | Mean F1 reranked |
|---|---|---|
| Random forest (final pipeline) | 0.461 | **0.474** |
| HistGradientBoosting | 0.460 | 0.468 |

No gain out of domain either. A stronger learner does not fix a domain gap that comes from the data.

**3. Few-shot adaptation to Vienna** (40% of Vienna held out for testing; *n* labelled listings drawn from the other 60%; F1-macro after reranking; `results/pilot_fewshot_vienna.csv`, `figures/extensions_pilot/fewshot_vienna.png`)

| Labelled Vienna listings *n* | 3 source cities + *n* Vienna | *n* Vienna only |
|---|---|---|
| 0 (pure domain generalisation) | 0.467 | – |
| 50 | 0.442 | 0.397 |
| 200 | 0.465 | 0.471 |
| 500 | 0.490 | 0.481 |
| 1,000 | 0.503 | 0.504 |
| 2,000 | 0.529 | 0.528 |
| all 6,069 | 0.564 | 0.564 |

The most striking result of the pilots:
* **About 200 labelled local listings are worth as much as 125,000 listings from three other cities.** From 500 labels on, local data alone beats pure domain generalisation, and from 1,000 labels the source cities no longer add anything.
* With 50 labels, adding Vienna as a fourth "balanced" domain *hurts* (0.442 < 0.467): each of the 50 listings gets the same total weight as all of Paris, so the forest overfits them. A smaller weight for the few target labels (or fine-tuning) is needed.
* For a real deployment in a new city, collecting a few hundred labelled listings is a better investment than more source cities. The few-shot curve is the right tool to size that labelling budget.
