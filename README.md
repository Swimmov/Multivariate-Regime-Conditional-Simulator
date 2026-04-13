# Multivariate Regime-Conditional Simulator
## Simulated World Project — Part 1: Building N Independent Synthetic Market Worlds

> *From 25 years of real market data to a library of N statistically faithful, structurally independent synthetic market universes - each indistinguishable from reality in distribution yet path-independent from every other world.*

---

## Project Goal

Generate **N independent synthetic market worlds**, each a complete daily time series of market factors and returns that:

- Matches the **marginal distributions** of real-world data
- Preserves **regime-conditional correlation structure** (calm / transition / stress)
- Reproduces **temporal memory** (AR(1) autocorrelation, volatility clustering)
- Is **statistically independent** from the real world and from every other simulated world (path correlation ≈ 0)
- Enables unlimited **intervention experiments, stress scenarios, and strategy validation** without touching live markets

---

## Architecture Overview

```
Part 1.1 — Data Collection & Feature Engineering
        │  Raw market + macro data → 58 engineered features → clean dataset
        ▼
Part 1.2 — Regime Detection & Factor Vetting
        │  GaussianHMM → 3 market states → Pro-Score factor ranking → robust factors
        ▼
Part 1.3 — Simulation Engine & World Library
           Gaussian Copula + β model + AR(1) + GARCH-lite → N × T synthetic worlds
```

---

## Part 1.1 — Data Collection & Feature Engineering

### 1. Source Data Collection

Two primary data sources, covering **2000-01-01 to 2026-01-01**:

**Yahoo Finance** — pulled via `yfinance`:

| Ticker | Variable | Role |
|--------|----------|------|
| `^GSPC` | S&P 500 OHLCV | Target price + volume |
| `^VIX` | CBOE Volatility Index | Fear gauge |
| `DX-Y.NYB` | US Dollar Index | FX market structure |

**FRED API** — two separate indicator groups:

*Group 1 — Daily financial rates:*

| Series | Description | Frequency |
|--------|-------------|-----------|
| `DGS10` | 10-Year Treasury Rate | Daily |
| `T10Y2Y` | 10Y minus 2Y Treasury spread | Daily |
| `DTB3` | 13-Week T-Bill rate | Daily |

*Group 2 — Economic releases (publication-lag-aware):*

| Series | Description | Release lag |
|--------|-------------|-------------|
| `GDP` | US Gross Domestic Product | ~4 weeks after quarter end |
| `CPIAUCSL` | Consumer Price Index | ~2 weeks after month end |
| `PAYEMS` | Non-farm Payrolls | ~1 week after month end (first Friday) |
| `UNRATE` | Unemployment Rate | ~1 week after month end |

> **Leakage prevention:** All FRED economic releases are merged using `pandas.merge_asof` with `direction='backward'`. Each trading day receives the most recently *published* (not most recently *observed*) value of each macro indicator, respecting real-world publication lags. `observation_period` and `as_of_date_release` columns are tracked separately for each indicator.

---

### 2. Feature Engineering

Features are built in four passes after data alignment:

**Pass 1 — Rate & Yield Curve Features** (from FRED financial data):
```python
term_spread      = DGS10 - T10Y2Y          # long-minus-short rate spread
T10Y3M           = DGS10 - DTB3            # 10Y vs 3M — classic recession predictor
T10Y2Y_yield_curve = (T10Y2Y < 0).astype(int)  # binary inversion flag
DGS10_dif        = DGS10.diff()            # daily rate change
T10Y2Y_dif       = T10Y2Y.diff()
DTB3_dif         = DTB3.diff()
```

**Pass 2 — Macro Release Features** (from FRED economic data):
```python
# Period indicators (seasonality signals)
period_CPI            = CPIAUCSL_observation_period.dt.month
period_GDP            = GDP_observation_period.dt.quarter
period_PAYMES_UNRATE  = PAYEMS_observation_period.dt.month

# Rate-of-change signals
GDP_pct()     = GDP_value.pct_change()
CPI_pct()     = CPIAUCSL_value.pct_change()
PAYEMS_pct()  = PAYEMS_value.pct_change()
UNRATE_pct()  = UNRATE_value.pct_change()

# Binary release-day event flags
is_GDP_release_day       # 1 on GDP announcement days
is_CPIAUCSL_release_day  # 1 on CPI announcement days
is_PAYEMS_release_day    # 1 on payrolls announcement days

# Composite kinetic factors
kin_f1_gdp_paym   = PAYEMS_value / GDP_value          # labor intensity ratio
kin_f2_unr_f1     = UNRATE_value/100 - kin_f1         # unemployment spread
kin_f3_cpi_t10y3m = CPI_pct()*100 - T10Y3M           # real-rate proxy
```

**Pass 3 — OHLC & Candlestick Features** (13 intraday structure features):
```python
# Price returns
Open_return  = Open.pct_change()
High_return  = High / Close.shift(1) - 1
Low_return   = Low / Close.shift(1) - 1
Close_return = Close.pct_change()

# Gap dynamics
gap        = (Open - Close.shift(1)) / Close.shift(1)
gap_filled = binary: whether intraday gap was closed

# Intraday volatility
daily_range      = (High - Low) / Close
range_pct_change = daily_range.pct_change()

# Candle structure
close_position   = (Close - Low) / (High - Low)  # [0,1]: 0=closed at low
body             = (Close - Open) / Close
upper_wick       = (High - max(Open,Close)) / Close
lower_wick       = (min(Open,Close) - Low) / Close
high_close_ratio = High / Close
low_close_ratio  = Low / Close
```

**Pass 4 — Technical & Momentum Features** (20 indicators via `ta` library):
```python
# Momentum oscillators
RSI_14_norm, RSI_7_norm                       # normalized to [-1, 1]

# Trend momentum
MACD_norm, MACD_signal_norm, MACD_diff_norm   # normalized

# Volatility regime
BB_width, BB_position, BB_squeeze

# Trend deviation
price_vs_EMA10, price_vs_EMA20
price_vs_SMA20, price_vs_SMA50
EMA_cross, MA_cross, trend_strength_ema

# Price momentum
ROC_10, ROC_20

# Trend strength
ADX_norm

# Volume-price
OBV_base_dif   # first difference of On-Balance Volume (stationary)
```

**Volume microstructure:**
```python
Volume_return  = Volume.pct_change()
volume_ratio   = Volume / Volume.rolling(20).mean()  # normalised for non-stationarity
volume_trend   = directional volume trend signal
```

---

### 3. Final Feature Set

```
Shape: (6,489, 59)  →  58 features + 1 target (Close_return)
Period: 2000-03-15 to 2025-12-31
NaN check: False (confirmed clean)
```

Features exported to: `features_for_part_1_2_df.csv`  
Close price exported to: `CLOSE_data_for_part_1_2.csv`

---

## Part 1.2 — Regime Detection & Factor Vetting

### 1. Data In

```python
data = pd.read_csv("project_DATA/features_for_part_1_2_df.csv",
                    index_col="Date", parse_dates=True)
# Shape: (6,489, 59) → after dropna: (6,468, 66)
```

---

### 2. Market State Variables

Five variables constructed to define the HMM observation space. All are leakage-free:

```python
log_ret      = np.log1p(return)                          # return shock
realized_vol = return.ewm(span=20).std().shift(1)        # lagged EWM volatility
VIX_level    = VIX_indx_close                            # fear level
VIX_change   = np.log(VIX_indx_close).diff()             # fear dynamics
vol_z        = ((Volume - Volume.rolling(20).mean()) /    # volume anomaly
                 Volume.rolling(20).std()).shift(1)
```

> `shift(1)` on `realized_vol` and `vol_z` ensures regime labels are computed from information fully available at decision time — zero look-ahead leakage.

---

### 3. Gaussian HMM Model

```python
model = GaussianHMM(
    n_components   = 3,        # calm / transition / stress
    covariance_type = "full",  # full covariance matrix per state
    n_iter         = 1000,
    random_state   = 42
)
model.fit(StandardScaler().fit_transform(hmm_features_df[state_cols]))
```

---

### 4. Regime Statistics — Within-Regime Mean Analysis

| Regime | Label | `log_ret` | `realized_vol` | `VIX_level` | `VIX_change` | `vol_z` |
|--------|-------|-----------|----------------|-------------|--------------|---------|
| 0 | **Calm** | +0.000716 | 0.00608 | 13.30 | −0.000108 | −0.022 |
| 1 | **Transition** | +0.000107 | 0.00972 | 19.76 | +0.000408 | +0.049 |
| 2 | **Stress** | −0.000405 | 0.01990 | 32.40 | −0.001195 | +0.065 |

The three states are economically interpretable: Calm = low-vol bull market; Transition = elevated uncertainty; Stress = crisis/bear market with VIX > 30.

Regime count:

| Regime | Days | % of sample |
|--------|------|-------------|
| 0 Calm | ~2,431 | 37.6% |
| 1 Transition | ~2,773 | 42.9% |
| 2 Stress | ~1,264 | 19.5% |

---

### 5. Real-World Transition Matrix — Creation and Save

```python
HMM_matrix = model.transmat_
np.save('project_DATA/HMM_real_world_transition_matrix', HMM_matrix)
```


**Transition Matrix** (high persistence, no direct calm↔stress jumps):
```
[[0.9795  0.0205  ~0    ]   <-- Calm: high persistence, no direct jump to stress
 [0.0180  0.9726  0.0094]   <-- Transition: moderate persistence, small stress entry
 [~0      0.0216  0.9784]]  <-- Stress: high persistence, exit only through transition
```
```
Calm->Calm: 0.980    Calm→Transition: 0.020    Calm->Stress: ~0
Trans->Calm: 0.018   Trans→Trans: 0.973        Trans->Stress: 0.009
Stress->Calm: ~0     Stress→Trans: 0.022       Stress->Stress: 0.978
```

Key property: **no direct calm <-> stress transitions** — markets must pass through the transition state, consistent with real crisis dynamics. The transition matrix is saved and reused in Part 1.3 to simulate regime sequences for synthetic worlds.

---

### 6. Close Price Visualisation by Regime

S&P 500 Close price scatter-coloured by HMM regime across selected historical windows:

- **2000–2002**: Dot-com bust - stress and transition dominate, no calm
- **2006–2008**: Bull market (calm) --> early deterioration (transition) --> crisis onset (stress) before price peak
- **2008–2010**: Financial crisis - stress through the trough, transition in recovery
- **2018–2020**: Late-cycle bull (calm/transition) -> COVID crash (stress)
- **2022–2024**: Rate-hike bear market (stress/transition) -> recovery (calm)

The model correctly flags stress **before** price peaks in several episodes, demonstrating that the HMM's volatility-based observation space leads price-level signals.

---

### 7. Factor Vetting — IC Metrics Table

The `vet_feature()` function computes 8 metrics for each of the 58 candidate features:

| Metric | Description |
|--------|-------------|
| `ic_all` | Spearman rank correlation vs `fwd_return` (overall) |
| `ic_calm` | IC within calm regime only |
| `ic_stress` | IC within stress regime only |
| `regime_gap` | `\|ic_calm − ic_stress\|` — regime instability flag |
| `vol_sensitivity` | IC difference: high-volume vs low-volume days |
| `stability_metric` | Rolling IC mean / rolling IC std — IC information ratio |
| `bootstrap_IC_mean` | Mean IC over 500 bootstrap resamples |
| `bootstrap_IC_interval` | 90% CI width — statistical reliability |

**Pro-Score formula** — composite multiplicative factor quality score:

```
pro_score = |IC|^1.0 × stability^1.5 × reliability^1.0 × regime_robustness^1.0 × liquidity^0.5

where:
  strength          = |bootstrap_IC_mean|^1.0
  stability         = stability_metric^1.5       ← highest exponent: stability prioritised
  reliability       = (1 / bootstrap_IC_interval)^1.0
  regime_robustness = (1 / (1 + regime_gap))^1.0
  liquidity         = (1 / (1 + vol_sensitivity))^0.5
```

The **multiplicative structure** means a factor scoring zero on any single dimension is eliminated entirely — there is no compensation between dimensions.

---

### 8. Filter and Penalties — Weak Factor Elimination Logic

Two boolean flags identify problematic factors:

**`ci_contains_zero`** — Bootstrap 90% CI includes zero:
- IC may be statistical noise
- The factor's predictive signal cannot be distinguished from zero with confidence

**`gap_gt_ic`** — Regime gap exceeds overall IC (`|ic_calm − ic_stress| > |ic_all|`):
- Regime dependence is stronger than the unconditional signal
- The factor "works" in one regime but not the other — conditional behaviour, not stable alpha

**Decision table:**

| `ci_contains_zero` | `gap_gt_ic` | Action |
|--------------------|-------------|--------|
| False | False | **KEEP** — strong and stable |
| True | False | **PENALIZE × 0.6** — weak evidence but stable |
| False | True | **PENALIZE × 0.4** — regime-dependent, borderline |
| True | True | **DROP** — no reliable signal |

Additional rule: factors where `ic_calm` and `ic_stress` have opposite signs are dropped — a factor that predicts positive returns in calm and negative returns in stress is not alpha, it is noise.

A `0.7×` penalty is applied to any factor where `regime_gap` is NaN (insufficient observations in one regime).

---

### 9. Output Data

| File | Description | Shape |
|------|-------------|-------|
| `_vet_features_df.csv` | Full feature dataset with `hmm_regime` labels | (6,468, 68) |
| `_pro_score_df_metrics_factors.csv` | Robust factors sorted by `pro_score_penalized` | (N_factors, 14) |
| `HMM_real_world_transition_matrix.npy` | 3×3 Markov transition matrix | (3, 3) |

---

## Part 1.3 — Simulation Engine & World Library

### 1. Data In

```python
vet_features_df       # full feature set + regime labels    (6,468, 68)
factor_metrics_sorted # robust factors sorted by pro_score  (14, 24)
trans_mat             # real-world HMM transition matrix    (3, 3)
```

Selected factors list (24 robust factors from Part 1.2 scoring):
`RSI_7_norm`, `RSI_14_norm`, `price_vs_EMA20`, `price_vs_SMA50`, `BB_position`, `price_vs_EMA10`, `price_vs_SMA20`, `high_close_ratio`, `VIX_indx_close`, `ROC_10`, `close_position`, `body`, `ROC_20`, `Low_return`, `news_sentiment`, `Open_return`, `upper_wick`, `kin_f3_cpi_t10y3m`, and others.

---

### 2. Regime Simulation

The real-world transition matrix drives a **first-order Markov chain** to generate synthetic regime sequences:

```python
def simulate_regimes(T, trans_mat, init_state=0):
    states = [init_state]
    for _ in range(T - 1):
        current = states[-1]
        next_state = np.random.choice(len(trans_mat), p=trans_mat[current])
        states.append(next_state)
    return pd.DataFrame(data=np.array(states), columns=['hmm_regime'])
```

Each call with a different random seed produces a unique, independent regime sequence with the same long-run statistical properties as the real world.

---

### 3. Regime Distribution Comparison: Real World vs Simulated

| Regime | Real World | Simulated | Sections (real) | Sections (sim) |
|--------|------------|-----------|-----------------|----------------|
| 0 Calm | 2,431 days | ~2,500 days | 46 episodes | ~44 episodes |
| 1 Transition | 2,773 days | ~2,550 days | 71 episodes | ~65 episodes |
| 2 Stress | 1,264 days | ~1,150 days | 26 episodes | ~22 episodes |

Counts vary by seed (Monte Carlo variance) but remain statistically consistent. Episode structure (number of separate regime spells and their duration distributions) matches the real-world pattern, confirming the Markov chain correctly reproduces persistence, not just frequencies.

---

### 4. Gaussian Copula — Fitting per Regime

> *"Any joint distribution can be decomposed into two separate pieces — the marginal distributions of each variable individually, and a copula that describes the dependence structure between them."*

The copula is fitted **separately within each regime**, capturing regime-conditional factor correlation:

**Step 1 — Rank-Gaussian transform** (Filliben correction):
```python
def rank_gauss(x):
    r = stats.rankdata(x) / (len(x) + 1)   # +1 avoids boundary singularities
    return stats.norm.ppf(r)
```

**Step 2 — Fit covariance matrix in Gaussian space per regime:**
```python
def fit_copula(df, features, regime_col='hmm_regime'):
    models = {}
    for r in sorted(df[regime_col].unique()):
        sub = df[df[regime_col] == r][features].dropna()
        Z = np.column_stack([rank_gauss(sub[c]) for c in features])
        Sigma = np.cov(Z.T)
        models[r] = {"cov": Sigma, "data": sub}
    return models
```

Fitting a separate `Sigma` per regime means the **inter-factor correlation structure changes correctly between calm and stress** — a property a single global copula would miss.

---

### 5. Factor Generation for Simulated Worlds

**Step 3 — Sample from the copula** (inverse ECDF makes marginals non-parametric):
```python
def sample_copula(models, features, regime, n):
    m = models[regime]
    Z = np.random.multivariate_normal(np.zeros(len(features)), m["cov"], size=n)
    U = stats.norm.cdf(Z)                               # back to uniform space
    X = np.zeros_like(U)
    for j, c in enumerate(features):
        sorted_vals = np.sort(m["data"][c])
        idx = (U[:, j] * (len(sorted_vals) - 1)).astype(int)
        X[:, j] = sorted_vals[idx]                      # inverse ECDF lookup
    return pd.DataFrame(data=X, columns=features)
```

The **inverse ECDF step** is what makes the simulator non-parametric: if `U=0.9`, it returns the 90th percentile value from the actual observed distribution of that feature — preserving fat tails, skewness, and any non-Gaussian shape without assumption.

Factors are generated separately for each regime and assembled in the order dictated by the simulated regime sequence.

---

### 6. Empirical Clipping — Bounds Enforcement

After factor generation, simulated values are clipped to empirical quantile bounds within each regime:

```python
def clip_to_empirical_by_regime(df_sim, df_real, factors):
    for regime in df_real['hmm_regime'].unique():
        lo_q = 0.001 if regime == 2 else 0.01   # wider bounds for stress
        hi_q = 0.999 if regime == 2 else 0.99
        for col in factors:
            lo = df_real[df_real['hmm_regime'] == regime][col].quantile(lo_q)
            hi = df_real[df_real['hmm_regime'] == regime][col].quantile(hi_q)
            regime_mask = df_sim['hmm_regime'] == regime
            df_sim.loc[regime_mask, col] = df_sim.loc[regime_mask, col].clip(lo, hi)
    return df_sim
```

Stress regime uses tighter extreme bounds (`0.1%–99.9%`) to preserve legitimate crisis-level tail behaviour. Clipping barely activates in practice (factor marginals are already well-calibrated by the copula), confirming the simulator is not over-generating extremes.

---

### 7. Factor Distribution Validation and Ljung–Box test. Real vs Simulated:

**Distribution**
`fwd_return` VS  `sim_return`

    count    6468.000000  6468.000000
    mean        0.000312   0.000217  
    std         0.012166   0.014901
    min        -0.119841  -0.126992
    25%        -0.004704  -0.005982
    50%         0.000645   0.000504
    75%         0.005860   0.006964
    max         0.115800   0.101665

`high_close_ratio` VS  sim`high_close_ratio`

    count    6468.000000   6468.000000
    mean        1.006087    1.005811
    std         0.007971    0.007442
    min         1.000000    1.000000
    25%         1.001192    1.001175
    50%         1.003283    1.003184
    75%         1.007976    1.007664
    max         1.104767    1.094833

---

**Ljung–Box test**
`Real data` "fwd_return":
    
    lag   lb_stat   lb_pvalue
    5 	78.754034 	1.529125e-15
    10 	107.486566 	1.712345e-18
    20 	171.892935 	3.712023e-26

#### *Very small p-values -> Strong rejection of “no autocorrelation”.*

`Simulated data` "sim_return":

    lag  lb_stat 	 lb_pvalue
    5 	78.586101 	1.657882e-15
    10 	84.485884 	6.591865e-14
    20 	102.168706 	5.143481e-13

#### *Simulated returns behave like the real series across multiple lags.*

#### Cross-factor correlation matrices are preserved within each regime, confirming the copula correctly captures the dependence structure, not just the marginals.

---

### 8. Return Simulation

The return engine implements a **regime-conditional structural model** with four components:

r_t = μ_r + β_r^T (F_t − μ_{F,r}) + φ_r · r_{t-1} + ε_{t,r}

ε_{t,r} ~ t_{df=8}(0, √h_t) => [Student-t fat-tailed noise] :

```Python 
eps = np.random.standard_t(df=df_t) * np.sqrt(h)
```

h_t = ω + α · ε²_{t-1} + β_g · h_{t-1} => [GARCH(1,1) volatility clustering]


| Component | Description | How estimated |
|-----------|-------------|---------------|
| `μ_r` | Regime mean return | Within-regime sample mean of target (`fwd_return`) |
| `β_r` | Factor loadings (slopes) | OLS on centered X and y within each regime |
| `φ_r` | AR(1) coefficient | Correlation of same-regime consecutive returns |
| `t(df=8)` | Tail shape | df=8 reduces kurtosis vs df=5 while preserving fat tails |
| `α = 0.05` | GARCH shock sensitivity | Calibrated to match real Ljung-Box statistics |
| `β_g = 0.90` | GARCH variance persistence | Standard equity vol clustering parameter |

**Regime transition handling:** GARCH conditional variance `h` is re-initialised to `sigma_r²` on each regime transition, preventing carry-over of one regime's variance into the next.

**AR(1) estimation detail:** `φ_r` is estimated using only pairs where both `r_t` and `r_{t-1}` belong to the same regime — cross-regime transitions are excluded from the AR estimate to avoid contamination.

---

### 9. Level Checks — `sim_return` vs `fwd_return`

| Property | Real (`fwd_return`) | Simulated (`sim_return`) | Status |
|----------|---------------------|--------------------------|--------|
| AR(1) autocorrelation | −0.10397 | ~−0.10722 | ✅ Near-exact |
| Regime vol ordering | calm < trans < stress | Preserved | ✅ |
| Calm regime vol | 0.00593 | ~0.007028 | ✅ +17% |
| Stress regime vol | 0.02172 | ~0.02653 | ✅ +21% |
| Path correlation (sim vs real) | — | −0.0262 | ✅ Independent |
| Ljung-Box lag 20 p-value | ~10⁻²⁶ | ~10⁻¹¹ | ⚠️ Weaker clustering |

The **path correlation of −0.0262** is the most important check: the simulator generates structurally similar but **path-independent** worlds. A correlation near zero confirms each synthetic world is a genuinely alternative market history, not a noisy copy of the real one.

Ljung-Box statistics are lower than real data (weaker serial clustering) — a known trade-off from using df=8. All p-values remain below 10⁻¹⁰, confirming serial dependence is still strongly present in all simulated worlds.

---

### 10. Simulated Worlds Pipeline — N_WORLDS

The full simulation is wrapped in a single `generate_world(seed)` function that produces one complete synthetic world per call:

```python
def generate_world(seed, vet_features_df, selected_factors_list,
                   betas, factor_means, return_means,
                   phi_dict, sigma_dict, trans_mat,
                   df_t=8, alpha=0.05, beta_g=0.90):
    np.random.seed(seed)
    T = len(vet_features_df)

    # 1. Simulate regime path using real-world transition matrix
    sim_regimes = simulate_regimes(T, trans_mat, init_state=0)

    # 2. Fit copula on real data (fixed — same for all worlds)
    models = fit_copula(vet_features_df, selected_factors_list)

    # 3. Sample factors from copula per regime
    # 4. Assemble in regime order
    # 5. Clip to empirical bounds
    # 6. Simulate returns with β model + AR(1) + GARCH
    df_world['world_id'] = seed
    return df_world
```

```python
N_WORLDS = 50

world_library = [
    generate_world(seed, ...) for seed in range(N_WORLDS)
]
all_worlds_df = pd.concat(world_library, ignore_index=True)
# Shape: (N_WORLDS × T, n_features + sim_return + world_id)
# Example: 50 worlds × 6,468 days = 323,400 rows
```

Each world differs only in its **random seed** — the structural parameters (betas, phi, sigma, transition matrix) are fixed and estimated from real data. This ensures all worlds share the same underlying causal structure while being statistically independent from each other.

---

### 11. Simulated Close Price Visualisation by Regime

Reconstructed Close price series from `sim_return` for each simulated world, scatter-coloured by regime label — mirroring the Part 1.2 real-world visualisation. Confirms that:

- Stress episodes cluster correctly (large drawdowns correspond to stress regime)
- Calm periods show low-volatility trending behaviour
- Transition periods exhibit the correct intermediate character

---

### 12. Realistic Close Price Generation sliced rows sample from 323,400 Virtual Days

From the full library of 50 worlds × 6,468 days = **323,400 simulated virtual trading days**, a representative sliced-days sample is drawn for visualisation:

```python
# Sample of sliced rows from the 323,400-row world library
all_data_viz(n_combined_sim_df.iloc[6500: 15000])
all_data_viz(n_combined_sim_df.iloc[11500:13000])
```

Close price is reconstructed from cumulative `sim_return`:

```python
close_sim = start_price * np.exp(np.cumsum(returns_sim))
```

The visualisation shows regime-coloured synthetic price paths that are:
- **Statistically indistinguishable** from real S&P 500 behaviour in distribution
- **Path-independent** from each other and from the real world
- **Structurally diverse** across the four transition matrix variants

---

## Repository Structure

```
SimulatedWorlds/
├── Part_1_1_Data_Collection_Feature_Engineering.ipynb
├── Part_1_2_Regime_Simulation_Factor_Score.ipynb
├── Part_1_3_Factors_Return_Simulation_Worlds_Library.ipynb
│
└── project_DATA/
    ├── YF_data_yfinance_i_sp500.csv
    ├── YF_data_yfinance_i_vix.csv
    ├── YF_data_yfinance_i_dxy.csv
    ├── data_fred_gdp_all_data.csv
    ├── data_fred_cpi_all_data.csv
    ├── data_fred_payems_all_data.csv
    ├── data_fred_unrate_all_data.csv
    ├── features_for_part_1_2_df.csv          <-- Part 1.1 output for Part 1.2 input
    ├── CLOSE_data_for_part_1_2.csv           <-- Part 1.1 output for Part 1.2 input
    ├── _vet_features_df.csv                  <-- Part 1.2 output for Part 1.3 input
    ├── _pro_score_df_metrics_factors.csv     <-- Part 1.2 output for Part 1.3 input
    └── HMM_real_world_transition_matrix.npy  <-- Part 1.2 output for Part 1.3 input
```
---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data collection | `yfinance`, `fredapi`, `requests` |
| Data processing | `pandas`, `numpy` |
| Machine learning | `sklearn` (StandardScaler, LinearRegression) |
| Regime detection | `hmmlearn.GaussianHMM` |
| Statistical methods | `scipy.stats` (rankdata, norm, t-distribution) |
| Visualisation | `matplotlib`, `seaborn` |

---
