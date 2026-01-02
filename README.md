
# Noise-Aware ML for High-Speed Measurement Correction

## Overview

This project looks at how machine learning can help **improve the accuracy of fast, noisy measurements**. In many experimental systems, measurements taken very quickly tend to be unreliable, but they are often used as-is. The idea here is to see whether ML can learn **systematic error patterns** related to measurement conditions and reduce overall error.

To keep the focus on methodology, the study uses a **synthetic, physics-inspired dataset** with known ground truth.

---

## What was done

* Simulated a high-speed measurement system with noise that increases as measurement time decreases
* Defined a baseline where raw measurements are treated as ground truth
* Trained ML models to correct measurement errors using noise-aware features
* Reformulated the task as **residual learning** (predicting correction instead of the signal)
* Used gradient boosting with noise-aware sample weighting

---

## Results

| Method                    | RMSE   |
| ------------------------- | ------ |
| No correction             | ~61    |
| ML correction             | ~51    |
| Noise-aware ML correction | ~40–45 |

The results show that ML can meaningfully reduce error when noise structure is taken into account.

---

## Project structure

```
dataset.py            # Data generation
rsme_nocorrection.py           # Baseline evaluation
ml_randomforest.py     #Randomforest ML model
ml_noiseaware.py   # Noise-aware ML model
measurement_data.csv  # Generated dataset
README.md
```

---

## Notes

* The dataset is synthetic and fully reproducible
* Filtering and weighting reflect realistic experimental constraints
* The goal is methodological understanding, not deployment

---

## Author

Aryan Verma
B.Tech, Chemical Engineering
IIT (BHU) Varanasi

---

## Disclaimer

This work uses synthetic data and is intended for research and educational purposes only.

  
