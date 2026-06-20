# -Postpartum-Risk-Predictor-PRP-A-Machine-Learning-Classification-System.ipynb
Postpartum Risk Predictor (PRP)

A machine learning system that flags maternal mental health risk from self-reported symptoms — built to catch what traditional screening tools miss.


Why this exists

Postpartum depression affects somewhere between 10% and 20% of new mothers worldwide, and yet it remains one of the most underdiagnosed conditions in maternal healthcare. The reason isn't a lack of awareness — it's that the standard screening tools depend entirely on a mother being willing and able to disclose how she's really feeling. Social stigma works against that disclosure at almost every turn.

This project tries to work around that limitation rather than fight it. Instead of relying on a single self-report answer, it looks at the pattern across eleven symptom dimensions — sleep, anxiety, guilt, bonding, appetite, and more — and uses that combined picture to flag risk even when individual answers are vague, hedged, or left blank entirely.

The dataset behind this project makes the problem visible in a very concrete way: 22.3% of respondents refused to answer the suicidality question, and 35.1% gave an ambiguous "maybe" on guilt. A binary screening form would record both as negatives. This pipeline doesn't.


What it actually does

Given a completed survey, the pipeline:


Cleans and deduplicates the raw response data
Converts categorical answers ("Yes", "Sometimes", "Two or more days a week"...) into a numeric severity scale
Sums those values into a single composite "depression score" per respondent
Applies a clinical threshold to assign a binary label — High-Risk or Low-Risk
Trains three different classifiers on that labeled data and compares how well each one performs
Surfaces which symptoms actually drive the prediction, using Random Forest feature importance


The output isn't meant to replace a clinician's judgment. It's meant to extend it — giving healthcare providers an early signal so they know which mothers might need a more careful follow-up conversation, well before the standard six-week postnatal check-up.


The dataset

Raw responses1,503Collection windowA single continuous 26-hour period (June 2022)Features11 (1 demographic, 9 symptom-based, 1 timestamp)Missing values27 cells (0.2%)Duplicate rows (after dropping timestamp)1,177Unique records used for modeling326Class split248 High-Risk (76.1%) / 78 Low-Risk (23.9%)

The heavy duplication is worth calling out honestly: once the timestamp column is removed, well over a thousand rows turn out to be symptomatically identical to another row. That's most likely the result of repeat submissions during the short collection window, and it's a key limitation discussed further down.

The eleven survey variables


Age (binned, 25–30 through 45–50)
Feeling sad or tearful
Irritable towards baby and partner
Trouble sleeping at night
Problems concentrating
Overeating or loss of appetite
Feeling anxious
Feeling of guilt
Problems bonding with baby
Suicide attempt
Timestamp (used only for deduplication, dropped before modeling)



How the risk label was built

There's no physician-confirmed diagnosis in this dataset, so a synthetic but clinically reasoned label had to be constructed.

Each categorical response is mapped to a numeric weight that reflects severity:

ResponseEncoded ValueNo0Maybe1Sometimes1.5Yes2Two or more days a week2.5Always3

These values are summed across all nine symptom columns (age is excluded) to produce a depression score. Any record scoring 7 or higher is labeled High-Risk; anything below that is Low-Risk. The threshold isn't arbitrary — it reflects the idea that PPD risk builds up across several concurrent symptoms rather than being triggered by any single answer.


Models and results

Three classifiers were trained on an 80/20 stratified split (260 training records, 66 test records) and evaluated with an emphasis on Recall, not raw accuracy — because in this context, a missed high-risk mother is a far more serious failure than an unnecessary follow-up call.

ModelAccuracyPrecisionRecallF1-ScoreLogistic Regression1.0001.0001.0001.000Random Forest0.9090.9070.9800.942SVC (RBF kernel)0.9550.9610.9800.970

A quick honesty note on that perfect Logistic Regression score: it's almost certainly an artifact of the composite scoring method on a small, clean test set, rather than a sign of a truly generalizable model — the classifier is essentially re-deriving the same threshold that was used to build the labels in the first place. Random Forest and SVC are the more trustworthy benchmarks here, and both maintain a 98% Recall, meaning only about 1 in 50 genuinely high-risk respondents would be missed.

What actually predicts risk

Random Forest's feature importance ranking turned up a few findings worth knowing before you dig into the notebook:


Sleep disruption and suicidality were the two strongest individual predictors — each contributing roughly 16% of total importance
Anxiety and guilt, despite being the most strongly correlated pair in the dataset (r = 0.547), ranked lower individually — when two features carry overlapping information, ensemble models tend to split credit between them rather than rewarding either one fully
Age contributed exactly zero predictive value. Risk in this cohort wasn't concentrated in any particular age bracket

What's inside the notebook, in order


Library imports and dataset load
Automated EDA report generation (ydata-profiling) → exports eda_report.html
Timestamp removal, missing-value check, and duplicate removal
Ordinal encoding of all categorical symptom columns
Composite depression score and binary label construction
Stratified train/test split
Training of Logistic Regression, Random Forest, and SVC
Evaluation — Accuracy, Precision, Recall, F1-Score, MSE — and a side-by-side comparison table



Honest limitations

This isn't a finished clinical product, and it shouldn't be treated as one. A few things worth keeping in mind before drawing conclusions from it:


The usable sample, after removing duplicate submissions, is only 326 records — collected within a single 26-hour window. That's a narrow, non-longitudinal snapshot, not a representative population sample.
The risk label is synthetically derived from a scoring formula, not from an actual clinical diagnosis. It's a reasonable proxy, not ground truth.
The dataset has no information on income, education, ethnicity, obstetric history, or prior psychiatric diagnosis — all of which meaningfully affect real-world PPD risk.
Self-reported data carries social desirability bias by nature. The model can only work with what people are willing to write down.



Where this could go next
Longitudinal tracking — moving from a single snapshot to repeated weekly check-ins, ideally modeled with an LSTM that can pick up on trajectory rather than a single point in time
Wider, cleaner data collection — a longer collection window with duplicate-prevention built in, plus richer demographic context
Multimodal signals — pairing self-report data with wearable sensor data (sleep tracking, heart rate variability) to reduce dependence on what people choose to disclose
Privacy-conscious deployment — a federated learning setup so models can train across institutions without centralizing sensitive patient data


Tech stack
Python · pandas · NumPy · scikit-learn · ydata-profiling · matplotlib · seaborn · Jupyter Notebook
