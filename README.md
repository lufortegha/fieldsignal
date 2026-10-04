# FieldSignal

**Small AI for context-aware agricultural attention under constrained connectivity**

FieldSignal is a lightweight agricultural attention system developed for the World Bank Small AI for Development Hackathon.

It uses historical and recent environmental data to identify when combinations of changing conditions become unusually different from local seasonal patterns and may deserve a farmer's attention.

FieldSignal does **not** diagnose crops or tell farmers how to farm. It surfaces measurable environmental evidence so farmers can combine it with their own knowledge and decide what to do next.

---

## The Problem

The hackathon agriculture scenario follows Noor, a smallholder farmer in the fictional Ondera highlands who grows coffee, maize, and beans.

Her constraints include:

- intermittent connectivity;
- purchasing 3G data only when needed;
- no home Wi-Fi;
- limited access to agricultural extension;
- a smartphone that often remains at home;
- limited ability to continuously retrieve and interpret technical environmental information.

The bottleneck FieldSignal addresses is therefore not simply access to another weather forecast.

It asks:

> **Have environmental conditions changed in a way that is unusual enough to deserve the farmer's attention?**

Instead of requiring the farmer to repeatedly check an application, FieldSignal is designed so environmental analysis can occur upstream and only meaningful changes need to be communicated.

---

## How FieldSignal Works

```text
NASA POWER environmental data
              ↓
     Data quality + freshness
              ↓
 Equivalent-season historical
           baseline
              ↓
 Environmental feature engine
              ↓
 Statistical context
        +
 Isolation Forest
              ↓
 Environmental Attention Score
              ↓
 ┌────────────┼─────────────────┐
 NORMAL    ATTENTION    INSUFFICIENT_EVIDENCE
              ↓
     Explainable evidence
              ↓
 Farmer UI / SMS-scale output
```

### Environmental features

The current prototype evaluates:

- accumulated rainfall;
- rainfall anomaly relative to the seasonal baseline;
- rainfall standardized anomaly;
- mean-temperature anomaly;
- mean-temperature standardized anomaly;
- maximum-temperature anomaly;
- maximum-temperature standardized anomaly.

Current conditions are compared with the **same calendar window in previous years**, rather than with an annual average.

---

## Small AI

FieldSignal combines two interpretable components:

1. **Statistical anomaly context** derived from standardized environmental variables.
2. **Isolation Forest**, a lightweight unsupervised machine-learning model that evaluates whether the combination of current environmental conditions resembles historical seasonal patterns.

The final attention layer combines this evidence into an Environmental Attention Score from 1–5.

The score represents **environmental unusualness and attention priority**.

It is **not** a probability of crop damage.

### Why not just fixed thresholds?

A controlled validation tested:

| Signal | Standardized value |
|---|---:|
| Rainfall | -1.5 |
| Mean temperature | -1.5 |
| Maximum temperature | -1.5 |

A simple rule:

```text
Alert if any |z| >= 2
```

would produce:

```text
NO ALERT
```

FieldSignal's existing multivariate model produced:

```text
Isolation Forest anomaly rank: 0.90
Environmental Attention Score: 4/5
State: ATTENTION
```

No individual signal crosses the simple ±2 threshold, but their **combination** is unusual relative to historical patterns.

This is a controlled model-validation case. It is not an observed farm condition and does not demonstrate crop damage.

---

## Real-Data Validation: Côte d'Ivoire

Ondera is fictional. For real environmental validation, FieldSignal uses a representative location in the **Man/Tonkpi coffee-growing context of Côte d'Ivoire**.

This location is not presented as Noor's actual farm.

### Verified 2026 run

**Representative coordinates:** `7.41, -7.55`

**Requested assessment:** October 4, 2026  
**Latest valid NASA POWER observations:** October 2, 2026  
**Data delay:** 2 days  
**Analysis window:** September 19–October 2, 2026  
**Historical comparison:** equivalent seasonal windows, 2016–2025

FieldSignal calculated:

| Measurement | Result |
|---|---:|
| Accumulated rainfall | 135.75 mm |
| Rainfall vs. seasonal baseline | **-21.6%** |
| Mean temperature vs. baseline | **+0.3°C** |
| Rainfall z-score | -0.55 |
| Mean-temperature z-score | +0.67 |
| Maximum-temperature z-score | +0.10 |
| Environmental Attention Score | **2/5** |
| State | **NORMAL** |
| Evidence-quality confidence | **HIGH** |

Despite rainfall being 21.6% below the seasonal mean, FieldSignal did **not** automatically issue an alert because the combined environmental pattern remained within the model's NORMAL attention range.

This demonstrates the purpose of contextualization: a changed environmental measurement does not automatically imply that farmer attention is warranted.

### Data freshness

NASA POWER did not yet contain valid values for October 3–4 at the time of testing.

FieldSignal therefore:

1. detected the latest valid date;
2. analyzed the latest complete 14-day window;
3. reported the actual analysis period;
4. disclosed a two-day data delay;
5. did not pretend the October 2 observations represented October 4.

Freshness is part of the evidence presented by the system.

---

## Designed for Constrained Connectivity

FieldSignal does not require the Small AI model to run on the farmer's phone.

The intended architecture is:

```text
Environmental data
        ↓
Upstream FieldSignal analysis
        ↓
Meaningful change?
     ↙       ↘
   NO        YES
   ↓          ↓
silence   small message
               ↓
        farmer's phone
```

This matters for users such as Noor who may have intermittent 3G access and purchase data only when necessary.

The web interface is primarily a prototype and evidence interface. A production deployment could communicate attention events through low-bandwidth channels such as SMS.

The hackathon prototype includes an **SMS-scale preview**, but it does not claim to operate a live SMS gateway.

---

## Localization

Scientific inference is language-neutral and represented as structured data.

The presentation layer can localize that result separately.

The prototype demonstrates:

- English;
- French.

Language should ultimately be **selected by the user**. Geography may suggest locally relevant options but should not determine a person's language preference.

Additional languages should only be introduced after appropriate local validation.

---

## Responsible AI

FieldSignal deliberately separates environmental evidence from agricultural authority.

### FieldSignal can

- detect unusual environmental patterns;
- compare recent conditions with historical seasonal patterns;
- prioritize potentially meaningful environmental changes;
- expose the measurements behind its assessment;
- disclose data quality and freshness;
- refuse assessment when evidence is insufficient.

### FieldSignal does not

- diagnose crop disease;
- determine the cause of crop symptoms;
- predict crop damage;
- predict yield loss;
- prescribe farming actions;
- replace farmers or agricultural extension professionals.

The intended decision relationship is:

```text
FieldSignal identifies a signal
              ↓
Farmer examines local conditions
              ↓
Farmer combines evidence with
local knowledge
              ↓
Farmer decides
```

---

## Evidence Quality and Safe Failure

FieldSignal distinguishes between:

```text
NORMAL
ATTENTION
INSUFFICIENT_EVIDENCE
```

Missing, malformed, or sufficiently unreliable environmental evidence does not silently become a valid assessment.

The prototype also tracks:

- missing-data fractions;
- historical baseline size;
- feature validity;
- requested analysis date;
- actual latest environmental-data date;
- data age;
- live/cached provenance.

`HIGH`, `MODERATE`, and `LOW` confidence refer to **evidence/data quality**, not the probability that a crop is stressed.

---

## Current Limitations

FieldSignal is a hackathon prototype.

Important limitations include:

- NASA POWER provides gridded environmental information, not a sensor measurement from Noor's field.
- Environmental anomaly does not establish crop impact.
- The current model has not been validated against observed coffee damage or yield outcomes.
- The 14-day analysis window is a prototype design choice rather than a field-validated optimum.
- Ten historical equivalent-season windows were used in the verified 2026 demonstration.
- Attention thresholds require broader field validation.
- The prototype does not send real SMS messages.
- The prototype has not yet been evaluated with farmers in Côte d'Ivoire.
- French localization demonstrates the architecture; it should not be assumed to be every farmer's preferred language.

---

## Validation Summary

| Validation | Status |
|---|---|
| NASA POWER ingestion | ✅ |
| Live 2026 environmental-data run | ✅ |
| Seasonal historical comparison | ✅ |
| Data-quality validation | ✅ |
| Data-freshness detection | ✅ |
| Lightweight multivariate ML | ✅ |
| Controlled comparison with simple threshold | ✅ |
| Explainable environmental evidence | ✅ |
| Safe failure / insufficient evidence | ✅ |
| English/French interface | ✅ |
| SMS-scale communication concept | ✅ |
| Actual SMS delivery | Not implemented |
| Agronomic outcome validation | Future work |
| Farmer field evaluation | Future work |

---

## Repository Structure

```text
fieldsignal_power.py
```

NASA POWER retrieval, caching, response parsing, and environmental-data validation.

```text
fieldsignal_features.py
```

Equivalent-season historical comparison and environmental feature engineering.

```text
fieldsignal_attention.py
```

Statistical baseline, Isolation Forest, Environmental Attention Score, confidence, evidence, and attention state.

```text
fieldsignal_api.py
```

FastAPI integration layer and demo endpoints.

Tests:

```text
test_fieldsignal_power.py
test_fieldsignal_features.py
test_fieldsignal_attention.py
test_fieldsignal_api.py
```

---

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest -v
```

Start the API:

```bash
python -m uvicorn fieldsignal_api:app --reload
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Data Source

Primary environmental source:

**NASA POWER — Prediction Of Worldwide Energy Resources**

FieldSignal uses NASA POWER daily meteorological data for environmental observations and historical seasonal comparison.

Any synthetic/demo fixtures included in the repository are explicitly demonstration data and must not be interpreted as observed farm conditions.

---

## Project Principle

> **FieldSignal does not tell the farmer how to farm. It helps identify when something outside the farmer's immediate field of view may deserve attention.**

**Farmer decides. Small AI helps determine when to look.**