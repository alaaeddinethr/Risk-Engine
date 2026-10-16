# 🏗️ Construction Project Risk Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://img.shields.io/github/actions/workflow/status/alaaeddinethr/construction-risk/ci.yml?style=flat-square&label=CI)

**Quantitative risk engine for construction projects — Monte Carlo schedule simulation,
Cost Overrun VaR, critical path analysis, and sensitivity analysis.**

[Live Demo](#) · [Documentation](#architecture) · [Quickstart](#quickstart)

</div>

---

## 📌 Context & Motivation

Construction projects are among the most risk-prone endeavors in engineering:
- **90%** of large projects experience cost overruns (McKinsey, 2017)
- Average delay: **+20 months** on major infrastructure projects
- Average cost overrun: **+80%** of original budget

This project applies **quantitative finance methods** (Monte Carlo, VaR, sensitivity analysis)
to construction project risk — a domain where these tools are standard in theory but
rarely implemented rigorously in practice.

Built on experience from a site engineering internship at **KA2C Construction, Casablanca (2025)**:
coordinating multi-trade execution, tracking progress, managing schedule constraints and HSE.

---

## 🔧 Core Methods

### Beta-PERT Distribution
Task durations are modelled with the Beta-PERT distribution, standard in project risk:
```
μ_PERT = (Optimistic + 4 × Most_Likely + Pessimistic) / 6
σ_PERT = (Pessimistic - Optimistic) / 6
```

### Monte Carlo Schedule Simulation
- Samples 10,000 project execution scenarios
- Each scenario draws independent PERT-distributed durations for all tasks
- Propagates through the dependency graph (CPM)
- Outputs: P10/P50/P90 completion dates, probability of meeting deadline

### Cost Overrun VaR
Applies Value-at-Risk methodology to project cost:
```
CostVaR_95 = P95(simulated_cost) − baseline_cost
```
The same framework used for financial portfolio risk — applied to construction budgets.

### Critical Path Analysis
- Deterministic CPM (Critical Path Method)
- Probabilistic criticality index: P(task is on critical path across all simulations)
- Schedule sensitivity: correlation of task duration with total project duration

---

## 🏗️ Architecture

```
construction-risk/
│
├── src/
│   ├── data/
│   │   └── project_generator.py   # Synthetic project generator (TCE, residential, infra)
│   │
│   ├── schedule/
│   │   ├── pert.py                # Beta-PERT distribution, 3-point estimates
│   │   └── cpm.py                 # Critical Path Method, float computation
│   │
│   ├── risk/
│   │   ├── monte_carlo.py         # Monte Carlo schedule + cost simulation engine
│   │   └── sensitivity.py         # Tornado chart, Spearman correlation analysis
│   │
│   └── dashboard/
│       └── app.py                 # Streamlit interactive dashboard
│
├── tests/
│   └── test_all.py
│
├── .github/workflows/ci.yml
├── config.yaml
└── requirements.txt
```

---

## 📊 Sample Results

On a simulated 18-task residential building project (TCE):

| Metric | Value |
|--------|-------|
| Baseline duration | 180 days |
| P50 completion | 198 days |
| P90 completion | 224 days |
| P(on time) | 18.4% |
| Baseline cost | €2,400,000 |
| P50 final cost | €2,574,000 |
| Cost VaR 95% | +€312,000 (+13.0%) |
| Top risk task | Structural works (criticality: 87%) |

---

## 🚀 Quick Start

```bash
git clone https://github.com/alaaeddinethr/construction-risk.git
cd construction-risk
pip install -r requirements.txt

# Generate a sample project
python src/data/project_generator.py --type residential --tasks 18

# Run Monte Carlo simulation
python src/risk/monte_carlo.py --project data/project.json --simulations 10000

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📚 References

- [1] PMI (2017). *A Guide to the Project Management Body of Knowledge (PMBOK® Guide)*. 6th Ed.
- [2] Vose, D. (2008). *Risk Analysis: A Quantitative Guide*. Wiley.
- [3] Bent Flyvbjerg (2014). *What You Should Know About Megaprojects*. Project Management Journal.
- [4] Malcolm, D.G. et al. (1959). *Application of a Technique for R&D Program Evaluation (PERT)*. Operations Research.

---

## 👤 Author

**Alaa Eddine TAHIR** — Engineering student at Arts et Métiers ParisTech
Site engineering internship at KA2C Construction, Casablanca (Jul–Aug 2025).
[LinkedIn](https://linkedin.com/in/alaaeddine-tahir) · [GitHub](https://github.com/alaaeddinethr)
