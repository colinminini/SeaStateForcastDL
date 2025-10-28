# Sea State Forecasting with Deep Learning and Hybrid Residual Modeling
**Colin Minini — CentraleSupélec & University College Dublin**  
**February – July 2025 — HIGHWAVE Project**  

---

### Overview
This research project explores **hybrid deep learning methods for forecasting ocean wave conditions** — in particular, the *significant wave height* (SWH) recorded by the **M6 buoy** off the west coast of Ireland.  
While physics-based numerical weather prediction (NWP) models provide reliable large-scale forecasts, they often exhibit **systematic local biases** and limited short-term accuracy.  
Here, we design deep neural networks that **learn the residuals between numerical forecasts and real observations**, effectively correcting physical model outputs through data-driven learning.

This work lies at the intersection of **scientific machine learning, time-series forecasting, and physical modeling** — bridging AI and oceanography in the context of the EU-funded **HIGHWAVE project**.

---

## Goals & Contributions
- **Reimplement and benchmark** state-of-the-art long-term time-series architectures (LSTM, TCN, PatchTST, SegRNN) on benchmark and real datasets.  
- **Build a robust forecasting pipeline** handling missing data and contiguous sliding-window sampling for marine time series.  
- **Propose a hybrid residual-learning framework** combining numerical forecasts (NOAA, ICON, MFWAM, etc.) with deep learning.  
- **Demonstrate measurable accuracy gains** over both standalone deep learning and raw physical forecasts.  

---

## Project Structure

├── data # Datasets created, processed and used for the project  
├── figures/ # All result plots (below)  
├── notebooks # The experimenting notebooks  
├── Sea_State_Forecast_Project_Report.pdf # Detailed technical report  
├── Sea_State_Forecasting_with_Deep_Learning_and_Hybrid_Residual_Modeling.ipynb # Main research notebook  

No installation or setup is required — this repository consists of a **single, self-contained Jupyter notebook** reproducing all experiments and figures.

---

## Datasets
| Dataset | Source | Resolution | Purpose |
|----------|---------|-------------|----------|
| **weather.csv** | Public meteorological dataset | 10 min | Model benchmarking and architecture testing |
| **M6 Observational** | Irish Marine Data Buoy Network | 1 h | Real-world univariate SWH forecasting |
| **Hybrid (M6 + Forecasts)** | NOAA, ICON, MFWAM, StormGlass API merge | 1 h | Hybrid DL + Numerical residual learning |

---

## Methodology
### 1. Forecasting formulation

For each 24-hour horizon \(H\), models use the past \(L = 336\) hours to predict the next 24 hours:

$$
f_\theta\left(X_{t-L:t}\right) \approx Y_{t+1:t+H}
$$

In the **hybrid** setup, models learn residuals:

$$
r_\theta(R_t) \approx Y - \hat{Y}_{\text{num}}, \qquad
\hat{Y} = \hat{Y}_{\text{num}} + r_\theta(R_t)
$$

### 2. Architectures Evaluated
- **LSTM** – Recurrent baseline for temporal dependencies  
- **TCN** – Causal dilated convolutions for sequence modeling  
- **PatchTST** – Transformer with patchwise attention for long-context forecasting [[Nie et al., 2023]](#references)  
- **SegRNN** – Segment Recurrent Neural Network optimized for long-term forecasting [[Lin et al., 2024]](#references)  
- **XGBoost** – Gradient-boosted tree baseline  

---

## Results

### Benchmark (weather.csv)
Multivariate long-range forecasting reproduces published SOTA results.  
SegRNN and PatchTST show the lowest errors and best temporal consistency.

| Model | MAE | MSE | Parameters | Train Time (s) |
|:------|----:|----:|------------:|---------------:|
| LSTM | 0.347 | 0.200 | 68 K | 0.84 |
| TCN | 0.370 | 0.233 | 55 K | 0.97 |
| PatchTST (uni) | 0.266 | 0.133 | 2.6 M | 36.4 |
| PatchTST (multi) | 0.269 | 0.206 | 2.6 M | 36.4 |
| SegRNN (uni) | 0.251 | 0.122 | 1.6 M | 5.98 |
| SegRNN (multi) | **0.227** | **0.187** | 1.6 M | 5.98 |

![Weather Forecast Example](figures/Weather_One_Sample_with_context.png)
![Weather Forecast Output](figures/Weather_One_Sample_just_Output.png)

---

### Hybrid Residual Learning

Physics-based numerical forecasts (e.g., NOAA, ICON, StormGlass) provide essential large-scale information but exhibit **systematic local biases** at the M6 buoy scale.  
To correct these biases, deep learning models were trained to **predict residuals** between observed and numerically forecasted significant wave height (SWH), effectively combining **physical priors with data-driven corrections**.

---

### Baseline: Physical Model Forecasts

| Rank | Model | MAE | MSE |
|:----:|:------|----:|----:|
| 1 | **NOAA (day 1)** | **0.425** | **0.338** |
| 2 | **StormGlass AI (day 1)** | 0.431 | 0.336 |
| 3 | **Meteo SG (day 1)** | 0.431 | 0.336 |
| 4 | ICON SG (day 1) | 0.442 | 0.409 |
| 5 | NOAA (day 2) | 0.561 | 0.568 |

The best physical forecasts reach **MAE ≈ 0.42 m** and **MSE ≈ 0.34**, forming the baseline for hybrid correction.

---

### Deep Learning Residual Models

| Rank | Model | MAE | MSE | Parameters | Train Time (s) |
|:----:|:------|----:|----:|------------:|---------------:|
| 1 | **LSTM** | **0.402** | **0.294** | 56.9 K | 0.21 |
| 2 | **SegRNN (uni)** | **0.406** | **0.297** | 1.59 M | 0.66 |
| 3 | **XGBoost** | 0.415 | 0.316 | 100 | 75.75 |
| 4 | **PatchTST (uni)** | 0.428 | 0.324 | 406 K | 0.96 |
| 5 | **TCN** | 0.444 | 0.353 | 43.9 K | 0.31 |

---

### 📈 Performance Summary

- The **hybrid LSTM and SegRNN models** achieve **MAE ≈ 0.40 m** and **MSE ≈ 0.29**, improving upon the best physical forecast (**NOAA day 1**, MAE = 0.425, MSE = 0.338) by approximately **5.6 % (MAE)** and **13 % (MSE)**.  
- Even lightweight architectures such as **LSTM** match or surpass the best physics-only baselines, confirming the value of **residual correction**.  
- **Tree-based XGBoost** remains competitive but less robust across time windows.  
- **Transformer-based PatchTST** yields stable performance with higher computational cost.  

![Residual Forecast Example](figures/Residuals_One_sample_with_context.png)

**Conclusion:**  
Hybrid residual learning effectively reduces systematic biases in physics-based ocean forecasts, demonstrating that **deep learning can serve as a statistical correction layer** for numerical wave models.

---

## Key Insights
- **LSTM and SegRNN** achieved the best overall MAE/MSE trade-off in both standalone and hybrid configurations.  
- **Residual learning** improved physical forecasts by up to **5–6 % in MAE** and **≈13 % in MSE** relative to the best numerical model.  
- **Multivariate pretraining** enhanced univariate forecasting through shared-weight generalization.  

---

## Future Directions
- Extend residual learning to multiple buoys with **spatial models (Graph NNs)**.  
- Introduce **uncertainty quantification** (e.g., Bayesian DL, quantile regression).  
- Explore **self-supervised pretraining** on large-scale meteorological archives.  
- Apply the hybrid correction paradigm to **climate, energy, and atmospheric** forecasting domains.

---

## References
- Lin S., Lin W., Wu W., Zhao F., Mo R., Zhang H. (2024). *Segment Recurrent Neural Network for Long-Term Time Series Forecasting.* (https://arxiv.org/abs/2308.11200)  
- Nie Y., Nguyen N. H., Sinthong P., Kalagnanam J. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers.*  (https://arxiv.org/abs/2211.14730)  
- Kong Y., Wang Z., Nie Y., Zhou T., Zohren S., Liang Y., Sun P., Wen Q. (2023). *Unlocking the Power of LSTM for Long-Term Time Series Forecasting.* (https://arxiv.org/abs/2408.10006)  
- Wen Q., Zhou T., Zhang C., Chen W., Ma Z., Yan J., Sun L. (2023). *Transformers in Time Series: A Survey.* (https://arxiv.org/abs/2202.07125)  
- Bai S., Kolter J. Z., Koltun V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* (https://arxiv.org/abs/1803.01271)

---

## 📬 Contact
For questions or collaborations: **colin.minini@student-cs.fr**
