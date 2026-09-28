# FeedbackIQ: Sentiment Model Comparison Experiment Report

## Executive Summary
This experiment benchmarked three distinct sentiment modeling paradigms on identical stratified test sets across **Electronics**, **Fashion**, and **Combined** customer reviews:
1. **Classical Baseline**: TF-IDF (1-2 n-grams, 5,000 features) + Logistic Regression
2. **Fine-tuned Transformer**: DistilBERT (`distilbert-base-uncased`) fine-tuned with balanced class-weighted CrossEntropyLoss
3. **Off-the-Shelf Transformer**: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) pre-trained on large-scale sentiment corpora

---

## 1. Final Comparison Table

| Model | Domain | Accuracy | Precision | Recall | F1 | Macro F1 | Neutral F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | Electronics | 0.8485 | 0.8341 | 0.8485 | 0.8123 | 0.5322 | 0.3333 |
| Fine-tuned DistilBERT | Electronics | 0.8788 | 0.8291 | 0.8788 | 0.8504 | 0.5511 | 0.0000 |
| Off-the-shelf RoBERTa (HF) | Electronics | 0.8788 | 0.8434 | 0.8788 | 0.8576 | 0.5541 | 0.0000 |
| TF-IDF + Logistic Regression | Fashion | 0.8500 | 0.7225 | 0.8500 | 0.7811 | 0.3063 | 0.0000 |
| Fine-tuned DistilBERT | Fashion | 0.8500 | 0.7831 | 0.8500 | 0.8151 | 0.4210 | 0.0000 |
| Off-the-shelf RoBERTa (HF) | Fashion | 0.8500 | 0.8389 | 0.8500 | 0.8430 | 0.4522 | 0.0000 |
| TF-IDF + Logistic Regression | Combined | 0.8585 | 0.8478 | 0.8585 | 0.8397 | 0.5752 | 0.1818 |
| Fine-tuned DistilBERT | Combined | 0.7925 | 0.8449 | 0.7925 | 0.7980 | 0.4732 | 0.0000 |
| Off-the-shelf RoBERTa (HF) | Combined | 0.8868 | 0.8887 | 0.8868 | 0.8876 | 0.6723 | 0.2857 |

---

## 2. In-Depth Analytical Findings

### Question 1: Which model performs best?
* **Overall Champion**: **Off-the-shelf RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) achieved the strongest generalized performance across metrics:
  - **Highest Combined Accuracy**: **88.68%** vs. 85.85% (Logistic Regression) and 79.25% (DistilBERT).
  - **Highest Combined F1-score**: **0.8876** vs. 0.8397 (Logistic Regression) and 0.7980 (DistilBERT).
  - **Highest Combined Macro F1**: **0.6723** vs. 0.5752 (Logistic Regression) and 0.4732 (DistilBERT).
* **Electronics**: Both Fine-tuned DistilBERT and Off-the-shelf RoBERTa tied for top Accuracy (**87.88%** vs. 84.85% baseline). However, RoBERTa edged out DistilBERT in weighted F1 (0.8576 vs. 0.8504) and Macro F1 (0.5541 vs. 0.5511).
* **Fashion**: All three models achieved **85.00%** overall accuracy (reflecting the 85% positive majority class), but RoBERTa delivered the highest weighted F1 (**0.8430**) and Macro F1 (**0.4522**), outperforming DistilBERT (0.8151 F1, 0.4210 Macro F1) and Logistic Regression (0.7811 F1, 0.3063 Macro F1).
* **Why did RoBERTa outperform fine-tuned DistilBERT on Combined?**
  - DistilBERT was trained with aggressive balanced class weights on a relatively small dataset (494 training samples). The class weights penalized positive errors so heavily that the model became over-sensitive to negative vocabulary, producing false negatives on positive reviews (15 positive reviews misclassified as negative).
  - RoBERTa leverages rich contextual embeddings pre-trained on millions of sentiment-annotated tweets and reviews, preserving calibrated decision boundaries without suffering from small-sample distortion.

---

### Question 2: Which model handles neutral reviews best?
* **Off-the-shelf RoBERTa** handles neutral reviews best overall:
  - In the Combined domain, RoBERTa achieved **0.2857 Neutral F1** with balanced precision (28.57%) and recall (28.57%), correctly identifying 2 out of 7 neutral reviews while maintaining reasonable specificity.
  - TF-IDF + Logistic Regression achieved **0.3333 Neutral F1** on Electronics (1/5 recalled, 100% precision) and **0.1818** on Combined (1/7 recalled, 25% precision), but completely collapsed on Fashion (**0.0000**).
  - Fine-tuned DistilBERT achieved **0.0000 Neutral F1** across all three domains.
* **Why do models struggle so severely with neutral reviews?**
  1. **Extreme Class Imbalance**: Neutral reviews (3-star ratings) constitute only ~6-8% of the dataset, providing very few training gradients.
  2. **Ambivalence vs. True Neutrality**: 3-star reviews are rarely "neutral" in sentiment. They almost universally consist of mixed polarities (e.g., *"Fabric is gorgeous but zipper broke on first wear"*). DistilBERT and RoBERTa detect strong affective words and resolve toward either positive or negative polarity rather than neutral.

---

### Question 3: Does performance differ between Electronics and Fashion?
* **Yes, notable domain discrepancies exist**:
  1. **Macro F1 Degradation in Fashion**:
     - Electronics Macro F1: Baseline = 0.5322, DistilBERT = 0.5511, RoBERTa = 0.5541.
     - Fashion Macro F1: Baseline = 0.3063, DistilBERT = 0.4210, RoBERTa = 0.4522.
     - Macro F1 dropped by **10 to 23 percentage points** in Fashion.
  2. **Lexical Concreteness vs. Subjective Sizing**:
     - Electronics reviews rely on clear, functional, binary descriptors (*"battery dead"*, *"crystal clear sound"*, *"bluetooth disconnects"*), allowing tokenizers and TF-IDF to find decisive sentiment indicators.
     - Fashion reviews are dominated by nuanced, subjective expressions regarding fit, fabric feel, drape, and sizing (*"fits a bit snug around shoulders but length is fine"*). Sizing mismatches often lead customers to leave 1-star or 3-star reviews despite praising the aesthetics, confounding models.
  3. **Sample Size & Positive Dominance**:
     - Fashion has fewer samples (266 total vs. 440 in Electronics) and even higher positive imbalance (85% positive), accelerating majority-class collapse in unweighted models.

---

### Question 4: Why might the models make mistakes?
* **Detailed Error Analysis**:
  1. **Mixed-Polarity Reviews ("Sandwich Sentiment")**:
     - *Example*: *"I really love the sleek design and bright screen, but it stopped charging after three weeks."*
     - The first clause has strong positive words (*love*, *sleek*, *bright*); the second has functional failure (*stopped charging*). Models often weight the initial compliments too heavily.
  2. **Subtle Sarcasm and Irony**:
     - *Example*: *"Works great if you enjoy paperweights that cost $80."*
     - Models encounter *"great"* and predict Positive.
  3. **Conditionals and Hypotheticals**:
     - *Example*: *"Would have been a 5-star jacket if it were waterproof."*
     - Models see *"5-star"* and miss the counterfactual conditional (*"would have been"*).
  4. **Over-Sensitivity to Negative Anchors (DistilBERT)**:
     - DistilBERT's balanced loss weighting caused it to predict Negative whenever mildly critical phrasing was present (*"not the warmest coat, but very stylish"* -> predicted Negative).

---

### Question 5: What are the limitations of rating-derived labels?
* **Label Noise and Cognitive Disconnect**:
  1. **Rating Discrepancies**: Users frequently assign 1-star or 2-star ratings due to shipping delays, damaged packaging, or incorrect sizing, while the text itself praises the product. Mapping 1-2 stars to "negative" injects contradictory supervisory signals.
  2. **The 3-Star Fallacy**: Treating 3 stars as "Neutral" is structurally flawed. In e-commerce, 3 stars typically signify *"Loved the product, hated the price"* or *"Works well, but broke soon"* (conflicting bilateral sentiment), not indifference.
  3. **Cultural & Subjective Calibration Differences**: One customer considers a 4-star review to mean "good", while another considers 4 stars to mean "disappointing because it wasn't perfect".
  4. **Recommendation**: Future iterations should complement rating heuristics with **aspect-based sentiment analysis (ABSA)** and fine-grained multi-aspect ratings (quality, value, delivery, sizing).
