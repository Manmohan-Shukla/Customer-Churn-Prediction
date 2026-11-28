Step 01 --- Problem Definition
ML Engineering Playbook | Structured-data machine learning

1. Objective
Translate a real-world business problem into a well-defined machine learning task with measurable success criteria. Do this before choosing algorithms or writing training code.

2. Why this step matters
A technically accurate model can still be useless if it predicts the wrong outcome, predicts too late, or cannot support a business decision. Problem definition determines what data to collect, what target to predict, how to evaluate the model and how predictions will be used.

3. Generic workflow
    . Identify the business problem and stakeholders.
    . Identify the decision that predictions will support.
    . Define the prediction unit (customer, transaction, product, etc.).
    . Define the target, observation window and prediction horizon.
    . Determine the ML problem type.
    . Identify available inputs and data constraints.
    . Choose evaluation metrics and a baseline.
    . Define expected output, success criteria and scope.
    . Document assumptions and verify feasibility.


4. Important concepts

  Concept                             Meaning

  Business objective                  Real-world outcome to improve

  ML objective                        Prediction task that supports that outcome

  Feature (X)                         Information available at prediction time

  Target (y)                          Outcome the model learns to predict

  Prediction unit                     Entity represented by one prediction

  Observation window                  Period from which features are collected

  Prediction horizon                  Future period for which the outcome is predicted

  Baseline                            Simple reference against which models are compared

  Data leakage                        Using information unavailable at prediction time


Choosing the problem type
  Requirement                       ML task                     Example
  Predict one of two labels         Binary classification       Customer churn

  Predict one of several labels     Multiclass classification   Product category

  Predict a continuous value        Regression                  House price

  Discover groups without labels    Clustering                  Customer segmentation

  Predict future values over time   Forecasting                 Monthly sales
Identify unusual observations     Anomaly detection           Unusual transactions
Not every business problem needs ML. Start with a simple rule or statistical baseline where appropriate.

5. Reusable problem definition template
Copy this section into every new project and fill it before implementation.
Project name:
Business problem:
Stakeholders:
Decision supported:
Business objective:
ML objective:
Prediction unit:
Target variable and label definition:
Observation window:
Prediction horizon:
Prediction frequency:
ML problem type:
Data source:
Available features at prediction time:
Expected model output:
Baseline:
Primary evaluation metric:
Secondary evaluation metrics:
Cost of prediction errors:
Success criteria:
Constraints and assumptions:
In scope:
Out of scope:
6. Project-specific implementation --- Customer Churn Prediction
Business problem
A telecom company loses customers when they discontinue their subscriptions. Customer attrition can reduce recurring revenue and increase the need for customer acquisition. The company wants to identify customers at risk of leaving so its retention team can prioritize outreach.
Business objective vs ML objective
- Business objective: reduce avoidable customer churn through targeted retention interventions.
- ML objective: estimate the probability that a customer belongs to the churn class using available customer information.
Predicting churn does not itself prove that an intervention will reduce churn. That requires separate business experimentation.
Problem specification
  Field                               Definition
  Stakeholder                         Telecom customer retention team

  Prediction unit                     Individual customer

  Problem type                        Supervised binary classification

  Target                              Churn

  Positive class                      Yes → 1 (churned)

  Negative class                      No → 0 (not churned)

  Data source                         Telco customer churn dataset, ingested into SQLite

  Candidate inputs                    Tenure, contract, monthly charges, services, payment method and demographics

  Identifier                          customerID --- useful for tracking, not a model feature

  Model output                        Estimated churn probability and threshold-based class
Business action                     Prioritize customers for retention outreach
Prediction horizon and limitation
A production use case could predict churn within the next 30 days using information available today. However, this is a proposed horizon, not a verified property of the current dataset. Without dated feature snapshots and outcome dates, the dataset supports retrospective classification; it cannot establish reliable prospective 30-day prediction performance. Verify dataset documentation and feature availability before making such a claim.

Example output (illustrative, not actual model predictions)
  Customer     Churn probability Predicted class at threshold 0.5
  A                         0.82 Churn
  B                         0.19 Not churn
  C                         0.64 Churn

The threshold is illustrative and should be selected using validation data, intervention costs and outreach capacity.
7. Evaluation strategy
  Metric                              What it measures
  Recall                              Fraction of actual churners correctly identified

  Precision                           Fraction of flagged customers who actually churn

  F1-score                            Harmonic mean of precision and recall

  ROC-AUC                             Ability to rank positive cases above negative cases

  PR-AUC / average precision          Precision-recall performance, especially useful for imbalanced data
  Accuracy                            Overall fraction of correct classifications

False positive: flag a customer who would not churn; potentially waste retention resources.

False negative: miss a customer who churns; potentially lose a retention opportunity.

Provisional primary metric: average precision for model comparison, with recall and precision at the chosen operating threshold. If missing churners is substantially more expensive, prioritize recall subject to an acceptable precision or outreach budget. Confirm this with stakeholders rather than assuming all false negatives cost more.

Baseline: use DummyClassifier (majority class for class predictions, class prior for probabilities), followed by Logistic Regression as a simple learned baseline.

Success criteria: outperform the baseline on held-out data and achieve an acceptable precision-recall tradeoff. No arbitrary numerical target is claimed before inspecting the data and business requirements. Business impact must be measured separately.

8. Scope and constraints
In scope: ingestion, cleaning, EDA, preprocessing, model training, evaluation, explainability and prediction API.

Out of scope for the initial version: automated retention campaigns, causal estimation of intervention effects and validated real-time production prediction.

Constraints: static dataset, possible class imbalance, unknown temporal feature availability, privacy and fairness considerations for demographic inputs.

9. Common mistakes and prevention
  Mistake                                   Prevention
  Selecting an algorithm first              Define the target and decision first

  Confusing business and ML objectives      Document both separately

  Using accuracy alone                      Inspect class balance and error costs

  Undefined prediction horizon              Specify when features and outcomes are observed

  Including post-outcome features           Audit feature availability at inference time

  Setting arbitrary performance targets     Establish baseline and business requirements
  Treating predictions as causal evidence   Evaluate interventions separately
10. Completion checklist
- [ ] Business problem and stakeholders identified
- [ ] Business decision and ML objective defined
- [ ] Prediction unit and target labels specified
- [ ] Problem type selected
- [ ] Observation window and prediction horizon documented or limitations acknowledged
- [ ] Data source and candidate features identified
- [ ] Baseline and evaluation metrics selected
- [ ] Prediction error costs considered
- [ ] Expected output, scope and constraints documented
- [ ] Data leakage risks identified

11. Interview questions
Why is churn prediction classification? The target is a categorical outcome with two possible classes.

Why not use accuracy alone? A majority-class model can have high accuracy while failing to identify churners.

Why predict probabilities instead of only labels? Probabilities enable ranking, prioritization and threshold adjustment; calibration should be assessed before interpreting them as reliable risks.

What is data leakage? Information unavailable at prediction time enters model development, inflating apparent performance.

How would you measure business success? Compare retention outcomes and net financial benefit against an appropriate control group, ideally through a randomized experiment.

12. Independent practice
Without AI, write a problem definition for loan default prediction using the reusable template. Then repeat for house price prediction. Explain why the first is classification and the second regression. Identify the prediction-time features and potential leakage in each case.

13. Note-taking and revision tips
- Keep generic concepts separate from dataset-specific decisions.
- Write the reason for each decision in your own words.
- Mark unverified assumptions explicitly and revisit them after EDA.
- Record the actual baseline and final metrics later; do not invent results.
- Close this file tomorrow and recreate the nine-step workflow from memory.
- Update the playbook whenever a new project exposes a limitation in your previous approach.

14. Key takeaways
Problem definition converts a business need into a measurable ML task. Define the target, prediction unit, timing, evaluation criteria and downstream decision before model development. A model is useful only when its predictions are valid for the intended use and support a meaningful decision.
