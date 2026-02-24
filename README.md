# 📊 Macro Regime Nowcaster

> Real-time economic regime detection using Dynamic Factor Models, Bayesian state-space methods (Kalman filter), and Markov-switching regime classification — with an LLM narrative agent and interactive Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              FRED API (60+ series)           │
                         └──────────────────┬──────────────────────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  DataPipeline   │
                                   │  - Transforms   │
                                   │  - Freq align   │
                                   │  - Ragged edge  │
                                   └────────┬────────┘
                                            │  Panel (T × N)
                              ┌─────────────▼─────────────┐
                              │   DynamicFactorModel (EM)  │
                              │   K=4 latent factors       │
                              └─────────────┬─────────────┘
                                            │  Factors (T × K)
                         ┌──────────────────▼──────────────────┐
                         │   RegimeSwitchingModel (Hamilton)    │
                         │   4 regimes: Expansion / Slowdown /  │
                         │   Recession / Recovery               │
                         └──────────────────┬──────────────────┘
                                            │  Regime probabilities
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
     ┌─────────▼──────────┐    ┌────────────▼────────────┐   ┌──────────▼────────┐
     │  RegimeAllocator   │    │   NarrativeAgent (LLM)  │   │  Streamlit        │
     │  Asset weights     │    │   + FedScraper          │   │  Dashboard        │
     └────────────────────┘    └─────────────────────────┘   └───────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/macro-regime-nowcaster.git
cd macro-regime-nowcaster
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add:
#   FRED_API_KEY=your_key_here       (required)
#   OPENAI_API_KEY=your_key_here     (optional, for LLM narrative)
```

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Fetch data

```bash
make fetch-data
# or: python scripts/fetch_data.py --start-date 2000-01-01
```

### 4. Run a nowcast

```bash
make nowcast
# or: python scripts/run_nowcast.py --output-format text
```

### 5. Launch the dashboard

```bash
make dashboard
# or: streamlit run src/dashboard/app.py
```

---

## Project Structure

```
macro-regime-nowcaster/
├── config/
│   ├── settings.yaml          # Model params, allocation weights, paths
│   └── fred_series.yaml       # 60+ FRED series with transforms & lags
├── src/
│   ├── data/                  # FRED client, pipeline, transforms, storage
│   ├── models/                # Kalman filter, DFM, regime switching, nowcaster
│   ├── allocation/            # Regime allocator, backtester
│   ├── agent/                 # Fed scraper, LLM narrative agent
│   ├── dashboard/             # Streamlit app
│   └── utils/                 # Logging, date utilities
├── tests/                     # pytest test suite
├── notebooks/                 # 5 Jupyter notebooks (EDA → full pipeline)
├── scripts/                   # CLI scripts (fetch, train, nowcast)
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## Configuration

### `config/settings.yaml`

Key sections:

| Section | Description |
|---------|-------------|
| `model.n_factors` | Number of latent factors (default: 4) |
| `model.n_regimes` | Number of economic regimes (default: 4) |
| `model.regime_labels` | Regime names list |
| `allocation.regime_conditional_weights` | Asset weights per regime |
| `data.start_date` | Historical sample start (default: 1980-01-01) |
| `agent.llm_model` | OpenAI model (default: gpt-4o-mini) |

### `config/fred_series.yaml`

Each entry specifies:
- `code`: FRED series ID (e.g. `PAYEMS`)
- `frequency`: `daily`, `weekly`, `monthly`, or `quarterly`
- `transform`: `log_diff`, `diff`, `pct_change`, `level`, or `none`
- `publication_lag_days`: Days after period end until release

---

## Model Methodology

### Kalman Filter / State-Space Model

The core estimation engine is a linear Gaussian state-space model:

```
State equation:        F_t = A × F_{t-1} + η_t,   η_t ~ N(0, Q)
Observation equation:  Y_t = C × F_t   + ε_t,     ε_t ~ N(0, R)
```

- **F_t** ∈ ℝᴷ: latent factors (K=4)
- **Y_t** ∈ ℝᴺ: observed economic indicators (N≈60)
- Missing observations (ragged edge) are handled by skipping the update step

### Dynamic Factor Model (EM Algorithm)

Parameters (A, C, Q, R) are estimated via the EM algorithm:
- **E-step**: Run Kalman smoother to compute sufficient statistics E[F_t | Y], E[F_tF_t' | Y]
- **M-step**: Update parameters via OLS closed-form solutions

Initialised with PCA for fast convergence.

### Hamilton (1989) Markov-Switching Model

Economic regimes follow a first-order Markov chain:

```
S_t | S_{t-1} ~ Categorical(P[S_{t-1}, :])
F_t | S_t = j ~ N(μ_j, σ²_j)
```

- **Filtered probabilities** via Hamilton (1989) forward recursion
- **Smoothed probabilities** via Kim (1994) backward smoother
- Transition matrix P estimated by EM

---

## Data Sources

| Category | Series | Source |
|----------|--------|--------|
| Labor Market | PAYEMS, UNRATE, ICSA, JTSJOL, U6RATE, … | FRED/BLS |
| Production | INDPRO, TCU, CFNAI, RSXFS, CMRMTSPL, … | FRED/Fed |
| Financial | T10Y2Y, BAA10Y, SP500, NFCI, STLFSI2, … | FRED |
| Consumer & Housing | UMCSENT, PERMIT, HOUST, HSN1F, … | FRED |
| Prices | CPIAUCSL, PCEPI, T5YIE, DCOILWTICO, … | FRED/BLS |
| Money & Credit | M2SL, TOTBKCR, BUSLOANS, … | FRED/Fed |
| GDP | GDPC1, GDPDEF, A261RX1Q020SBEA | FRED/BEA |

All data from the [Federal Reserve Bank of St. Louis FRED database](https://fred.stlouisfed.org/).

---

## Academic References

1. Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357–384.
2. Kim, C.J. (1994). *Dynamic Linear Models with Markov-Switching*. Journal of Econometrics, 60(1-2), 1–22.
3. Doz, C., Giannone, D., & Reichlin, L. (2012). *A Quasi Maximum Likelihood Approach for Large Approximate Dynamic Factor Models*. Review of Economics and Statistics, 94(4), 1014–1024.
4. Bańbura, M., & Rünstler, G. (2011). *A Look into the Factor Model Black Box*. International Journal of Forecasting, 27(2), 333–346.
5. Stock, J.H., & Watson, M.W. (2002). *Forecasting Using Principal Components from a Large Number of Predictors*. Journal of the American Statistical Association, 97(460), 1167–1179.

---

## Development

### Running Tests

```bash
make test
# or: pytest tests/ -v --cov=src
```

### Linting & Formatting

```bash
make lint     # ruff + mypy
make format   # black + ruff --fix
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install package in editable mode |
| `make install-dev` | Install with dev dependencies |
| `make test` | Run test suite with coverage |
| `make lint` | Run ruff and mypy |
| `make format` | Auto-format with black and ruff |
| `make fetch-data` | Fetch/update all FRED data |
| `make train` | Train models on historical data |
| `make nowcast` | Run a single nowcast |
| `make dashboard` | Launch Streamlit dashboard |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Run `make lint && make test`
5. Open a Pull Request

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

---

## License

MIT License — see [LICENSE](LICENSE) for details.