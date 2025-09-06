AML Name Screening

Transliteration-aware name screening for Anti-Money Laundering (AML)
Stack: Streamlit UI · XGBoost (primary) · n8n alerts · Fuzzy name matching

What it does

Input: Arabic or Latin full name

Process: transliterate → fuzzy-match against watchlist → combine with Risk Category

Output: ALLOWED · REVIEW · BLOCKED, with similarity %, top match, and XGBoost risk %

Alerts: n8n sends an email for REVIEW/BLOCKED

How matching works (short)

Transliteration: Arabic ⇄ Latin (normalize accents, spacing, common prefixes)

Fuzzy similarity (0–1):
Jaro-Winkler (overlap), Levenshtein (normalized edits), Soundex (phonetic 1/0)
→ Composite = weighted blend of the three

Models (brief)

XGBoost — primary: boosted trees with best generalization in tests

Random Forest — benchmark: bagged trees baseline

Logistic Regression — benchmark: linear baseline

Models train once (first run), then load from disk next runs.

Project layout
backend/
  model.py                 # AMLNameMatcher: features, training, matching
  models/                  # saved artifacts (auto-created)
  cleaned_aml_data.xlsx    # local dataset (not committed)
front/
  app.py                   # Streamlit UI (uses AMLNameMatcher)
n8n/
  workflows/aml_alert_webhook.json  # exported workflow (optional)

Run (quick start)
Python / Streamlit
# Linux/macOS
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r front/requirements.txt
export N8N_WEBHOOK_URL="http://localhost:5678/webhook/aml/alert"   # optional (for alerts)
cd front && streamlit run app.py

# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate
pip install -r backend\requirements.txt
pip install -r front\requirements.txt
$env:N8N_WEBHOOK_URL = "http://localhost:5678/webhook/aml/alert"   # optional (for alerts)
cd front; streamlit run app.py

n8n (email alerts)

Start locally (Docker):

docker run -it --name n8n -p 5678:5678 n8nio/n8n:latest


Workflow (minimal):

Webhook (POST) → Path: aml/alert → Response: Last Node

IF (use Expression):

{{ ["REVIEW","BLOCKED"].includes((($json.body && $json.body.decision) ? $json.body.decision : ($json.decision || "")).toString().toUpperCase().trim()) }}


Send Email (on true) — configure SMTP and Check Connection

Respond to Webhook (on both branches):

true body:

{"status":"email_sent","decision":"{{ ($json.body && $json.body.decision) ? $json.body.decision : $json.decision }}"}


false body:

{"status":"autorisé","decision":"{{ ($json.body && $json.body.decision) ? $json.body.decision : $json.decision }}"}


Test URLs

Prod (workflow Active): /webhook/aml/alert

Test (Listen for test event): /webhook-test/aml/alert (click Listen before sending)

App → n8n payload (example)
{
  "input": "yasmine bahri",
  "decision": "REVIEW",
  "similarity": 88.34,
  "confidence": 91.2,
  "reason": "High similarity (88.3%) with PEP",
  "xgbRiskPct": 73.12,
  "match": {
    "full_name": "يسري الدباغ",
    "risk_category": "PEP",
    "nationality": "France",
    "notes": "Former diplomat"
  }
}

Notes

Email only for REVIEW/BLOCKED (ALLOWED → no email)

To see runs in n8n: Workflow Settings → Save Successful Executions = All

Do not commit real datasets, model artifacts with sensitive data, or secrets.

Set N8N_WEBHOOK_URL in the terminal that runs Streamlit to enable alerts.

One-line description

AML name screening with transliteration-aware matching, XGBoost risk scoring, Streamlit UI, and n8n email alerts.
