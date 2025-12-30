# Mini-Project: Market Basket Analysis and Product Recommendations
## Apriori and FP-Growth Algorithms on Real Data

---

## 1. Context

An e-commerce company provides sales data for analysis. Each line in the file corresponds to a product sold in a transaction (invoice). The project objective is to identify item associations (rules of the type: IF a customer buys X, THEN they often buy Y) to propose product recommendations (cross-selling, bundles, store layout, etc.).

You will mainly use:
- The Apriori algorithm
- The FP-Growth algorithm
- Association rule generation functions (support, confidence, lift)

---

## 2. Learning Objectives

At the end of this mini-project, the student should be able to:

1. Prepare a transactional dataset from raw data (CSV/Excel files)
2. Apply Apriori and FP-Growth to extract frequent itemsets and association rules
3. Correctly interpret metrics: support, confidence, lift
4. Translate these rules into concrete business recommendations (combo offers, product placement, "You might also like...")
5. Briefly compare Apriori vs FP-Growth (computation time, number of rules, readability)

---

## 3. Data and Environment

### Possible Datasets (use at least one):
- **order_data.csv**: Small demonstration dataset (a few dozen transactions)
- **Online Retail.xlsx** or **transaction_dataset.csv**: Real data with several thousand transactions

### Working Environment:
- Python (Jupyter Notebook / Google Colab)
- Libraries: `pandas`, `numpy`, `matplotlib` or `seaborn` (optional)
- `apyori` or `mlxtend.frequent_patterns` for Apriori
- `mlxtend.frequent_patterns` for FP-Growth

---

## 4. Work to Complete

### Part 0 – Getting Started (Apriori on Small Dataset)

1. Load the `order_data.csv` file
2. Build a list of lists containing, for each transaction, the list of purchased items
3. Apply the Apriori algorithm (e.g., with `apyori`) using parameters suggested by the instructor, for example:
   - `min_support = 0.25`
   - `min_confidence = 0.2`
   - `min_lift = 2`
   - `min_length = 2`
4. Display several association rules (3 to 5 rules) and, for at least two of them:
   - Write the rule in the form: {X} ⇒ {Y}
   - Interpret support, confidence, lift in simple (business) language

**Objective of Part 0**: Refresh understanding of how Apriori works and the meaning of metrics.

---

### Part 1 – Real Data Preparation

1. From the `Online Retail.xlsx` or `transaction_dataset.csv` file:
   - Keep only useful columns (e.g., invoice ID, product ID, quantity, possibly country or date)
   - Clean the data: remove missing values, inconsistent rows, etc.

2. Build the transactional dataset:
   - Each transaction = one invoice
   - Each transaction = list of products purchased in that invoice

3. In the report, indicate:
   - Total number of transactions
   - Number of different items
   - Provide 3 examples of transactions (lists of items)

---

### Part 2 – Rule Extraction with FP-Growth

1. **Encode transactions:**
   - Use `TransactionEncoder` or mlxtend functions to transform the list of lists into a boolean table (one-hot)
   - Obtain a DataFrame where each column corresponds to an item and each row to a transaction (True/False)

2. **Apply FP-Growth:**
   - Choose a reasonable `min_support` value (e.g., 0.02 or 0.03, adjust according to dataset size)
   - List the 10 frequent itemsets with the highest support

3. **Generate association rules:**
   - Use `association_rules` (mlxtend) with `metric="confidence"` and a `min_threshold` value (e.g., 0.5 or 0.6)
   - Sort rules by confidence or lift

4. **Select 5 interesting rules:**
   - Support neither too low nor too high
   - High confidence
   - Lift > 1 (ideally > 1.2)

For each selected rule, you must:
- Rewrite it in plain language (e.g., "Customers who buy A and B often buy C")
- Explain what support, confidence, lift mean for this rule
- Propose at least one marketing action (bundle, promotion, store arrangement, online recommendation, etc.)

---

### Part 3 – Apriori vs FP-Growth Comparison

1. Apply Apriori on the same real dataset (or a subsample) with compatible parameters (`min_support`, etc.)

2. Compare Apriori and FP-Growth on:
   - Computation time (approximate, e.g., measure with `%time`)
   - Number of frequent itemsets and/or generated rules
   - Readability of results (too many rules, redundant rules, etc.)

3. In the report conclusion, answer the following questions:
   - Which algorithm (Apriori or FP-Growth) would you recommend for this type of data, and why?
   - What are, in your opinion, the limitations of association rules on this dataset?
   - Suggest two improvement paths (segmentation by country, by period, filtering certain items, item graph visualization, etc.)

---

## 5. Expected Deliverables

### 1. Jupyter Notebook (.ipynb) containing:
- Data preparation code
- Apriori code
- FP-Growth code
- Rule extraction and selection
- Some simple graphics (optional): top products, support distribution, etc.
- Markdown cells explaining your choices and results

### 2. Synthetic Report (3-4 pages) in PDF format, including:
- Introduction (context and objective of the mini-project)
- Methodology (data preparation, parameter choices, description of algorithms used)
- Main results (tables or screenshots of selected rules with interpretation)
- Business recommendations for the company
- Conclusion including Apriori/FP-Growth comparison

---

## 6. Adaptation / Enhancement (Master's Version)

### A. Learning Objectives (Master's Version)

1. Configure and justify hyperparameters (min_support, min_confidence, sorting metric, min/max itemset size) based on theoretical and empirical arguments
2. Conduct sensitivity analysis of these hyperparameters on:
   - Number of itemsets
   - Type of rules obtained
   - Business value of rules
3. Segment the dataset (by country, period, customer type, etc.) and compare rules obtained in each segment
4. Discuss limitations of association rules (correlation vs causation, popularity bias, noise, seasonality, etc.)
5. Propose a draft evaluation protocol if these rules were used in a recommendation system (A/B test, offline metrics...)

### B. Required Extensions (to add to Part 2 or 3)

#### 1. Sensitivity Analysis (mandatory)

Vary `min_support` (e.g., 0.01, 0.02, 0.03, 0.05) and observe:
- Number of frequent itemsets
- Number of generated rules
- Rule profile (trivial rules vs rarer but more informative rules)

Present results in a table or graph and comment.

#### 2. Dataset Segmentation (mandatory)

On the real dataset, choose a segmentation criterion (for example):
- A country (UK vs others)
- A period (high/low season)
- A product category

For at least two segments, repeat:
- Frequent itemset extraction
- Association rule generation
- Selection of 3 to 5 interesting rules per segment

In the report, compare segments:
- Which rules are common?
- Which rules are specific to a segment?
- What different business recommendations can be proposed according to segments?

#### 3. Link to Recommendation Systems (mandatory)

Write a short section (~1 page) explaining:
- How these association rules could be integrated into a recommendation system (e.g., rules used for a "Frequently bought together" module)
- What relevant evaluation metrics would be: click-through rate, conversion rate, average cart value, etc.
- What experimental protocol (A/B testing, offline metrics) the company could implement

#### 4. Critical Discussion (mandatory)

Include a more critical discussion on:
- Methodological limitations (correlation vs causation, missing data, seasonal bias)
- Potential risks of misusing rules (over-personalization, over-exposure of certain products, effects on sales diversity)
- Paths for coupling with other approaches (collaborative filtering, sequential models, etc.)

---

## C. Indicative Grading Rubric

- **Data preparation and notebook clarity**: 20%
- **Correct application of algorithms (Apriori, FP-Growth)**: 20%
- **Sensitivity analysis and segmentation**: 25%
- **Business interpretation and recommendations**: 20%
- **Report quality (structure, figures, critical thinking, bibliography if applicable)**: 15%

---

## Key Parameters Reference

### Apriori Parameters (Part 0):
```python
min_support = 0.25
min_confidence = 0.2
min_lift = 2
min_length = 2
```

### FP-Growth Parameters (Part 2):
```python
min_support = 0.02  # or 0.03, adjust according to dataset
min_threshold = 0.5  # or 0.6 for confidence
```

### Rule Selection Criteria:
- Support: neither too low nor too high
- Confidence: high
- Lift: > 1 (ideally > 1.2)

---

## Data Files

- `order_data.csv` - Small demo dataset
- `Online Retail.xlsx` - Real retail data
- `transaction_dataset.csv` - Alternative real dataset

---

## Libraries to Use

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # optional
import seaborn as sns  # optional

# For Apriori
from apyori import apriori  # or use mlxtend

# For FP-Growth and rule generation
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
```