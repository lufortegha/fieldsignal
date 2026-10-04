\# FieldSignal



FieldSignal is a Small AI prototype for context-aware agricultural attention under constrained connectivity.



\## Hackathon Scenario



The prototype addresses the agriculture scenario of the World Bank Small AI for Development Hackathon.



FieldSignal helps identify when changing environmental conditions become unusual enough to deserve a smallholder farmer's attention.



The current demonstration focuses on coffee farming.



\## What FieldSignal Does



FieldSignal combines:



1\. NASA POWER environmental data

2\. equivalent-season historical baselines

3\. environmental feature engineering

4\. lightweight multivariate anomaly detection

5\. deterministic attention logic

6\. explicit data-quality confidence



The system returns one of:



\- `NORMAL`

\- `ATTENTION`

\- `INSUFFICIENT\_EVIDENCE`



\## What FieldSignal Does Not Do



FieldSignal does not diagnose crop disease, predict crop damage, predict yield loss, or replace farmer or agricultural expert judgment.



Environmental anomaly is not equivalent to crop damage.



\## Small AI



The prototype uses a lightweight Isolation Forest alongside transparent statistical environmental features.



Generative AI is not required for core inference.



\## Run



Install dependencies:



```bash

python -m pip install -r requirements.txt

