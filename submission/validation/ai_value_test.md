\# Controlled Small-AI Validation



This test evaluates whether FieldSignal's multivariate anomaly model can identify a historically unusual combination that a simple univariate threshold would miss.



\## Controlled Input



| Feature | z-score |

|---|---:|

| Rainfall | -1.5 |

| Mean temperature | -1.5 |

| Maximum temperature | -1.5 |



\## Simple Baseline



Rule:



`Alert if any abs(z) >= 2`



Result:



`NO ALERT`



\## FieldSignal Result



\- Statistical baseline score: 1.5

\- Isolation Forest raw score: -0.58746

\- Empirical anomaly rank: 0.90

\- Environmental Attention Score: 4/5

\- Final state: `ATTENTION`



\## Interpretation



No individual environmental feature crosses the simple ±2 threshold.



FieldSignal identifies their combined multivariate pattern as unusual relative to the historical seasonal distribution.



This is a controlled model-validation case. It is not an observed farm condition, crop diagnosis, crop-damage prediction, or evidence of improved agricultural outcomes.



No production model parameters were changed to generate this test.

