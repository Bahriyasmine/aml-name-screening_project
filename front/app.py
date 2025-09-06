# app.py
# Run with:  streamlit run app.py
import os, sys, time
import pandas as pd
import numpy as np
import streamlit as st
import requests  # ← n8n webhook

# ---------- Robust import of AMLNameMatcher from model.py ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

def _import_matcher():
    # 1) Try regular import if model.py is already on sys.path
    try:
        from model import AMLNameMatcher  # type: ignore
        return AMLNameMatcher
    except Exception:
        pass

    # 2) Try to load by file path (front/model.py or parent/model.py)
    import importlib.util
    candidates = [
        os.path.join(HERE, "model.py"),
        os.path.join(ROOT, "model.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("model", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["model"] = module
            spec.loader.exec_module(module)
            return module.AMLNameMatcher

    raise ModuleNotFoundError(
        f"Could not locate model.py at {candidates[0]} or {candidates[1]}"
    )

AMLNameMatcher = _import_matcher()
# ------------------------------------------------------------------

# ---------- n8n webhook helper ----------
# FIX: URL configurée directement au lieu de dépendre de la variable d'environnement
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/aml/alert").strip()

def notify_n8n(payload: dict):
    """POST to n8n webhook if URL is configured. Non-blocking on failure."""
    if not N8N_WEBHOOK_URL:
        print("[n8n] No webhook URL configured")
        return
    
    try:
        print(f"[n8n] Sending payload: {payload}")
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        print(f"[n8n] Response: {response.status_code} - {response.text[:200]}")
        print("[n8n] Webhook sent successfully")
            
    except Exception as e:
        # Ne bloque jamais l'UX si n8n est down / URL invalide
        print(f"[n8n] notify failed: {e}")
# ----------------------------------------

# ---------- Caching ----------
@st.cache_resource(show_spinner=False)
def get_matcher(dataset_path: str):
    """
    Create/load the matcher. If saved models are missing, train once and cache.
    """
    m = AMLNameMatcher(dataset_path)
    data_ok = m.load_and_prepare_data()

    loaded = False
    try:
        loaded = m.load_models()
    except Exception:
        loaded = False

    if not loaded:
        if not data_ok:
            raise RuntimeError("Failed to load data and no saved models found.")
        with st.spinner("Training models (first run only)…"):
            m.train_models(test_size=0.30, use_cross_validation=False)
            m.save_models()
    return m


@st.cache_data(show_spinner=False)
def format_history_df(hist_rows):
    cols = ["Time", "Input", "Decision", "Similarity %", "Risk Category", "XGBoost Risk %"]
    if not hist_rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(hist_rows)
    return df[cols]


# ---------- UI ----------
st.set_page_config(
    page_title="AML Name Screening",
    page_icon="🛡️",
    layout="centered"
)

with st.sidebar:
    st.title("⚙️ Settings")
    dataset_path = st.text_input(
        "Dataset path",
        value="cleaned_aml_data.xlsx",
        help="Excel/CSV used by the matcher and models."
    )
    threshold = st.slider(
        "Match threshold (composite)",
        min_value=0.50, max_value=0.90, value=0.70, step=0.01,
        help="Minimum composite similarity to accept a candidate as a match."
    )
    show_debug = st.toggle(
        "Show transliteration & canonical forms",
        value=False
    )
    st.caption("Risk probability shown below uses your **XGBoost** model.")

    # --- n8n debug panel (sidebar) ---
    st.divider()
    st.caption("🔌 n8n webhook")
    st.code(N8N_WEBHOOK_URL if N8N_WEBHOOK_URL else "(No webhook URL configured)", language="text")
    
    if st.button("Test n8n webhook (send REVIEW)"):
        if not N8N_WEBHOOK_URL:
            st.error("N8N_WEBHOOK_URL not configured.")
        else:
            try:
                test_payload = {
                    "decision": "REVIEW", 
                    "input": "test_name",
                    "similarity": 88.5,
                    "confidence": 75.2,
                    "reason": "Test webhook",
                    "xgbRiskPct": 65.0
                }
                r = requests.post(N8N_WEBHOOK_URL, json=test_payload, timeout=10)
                st.success(f"n8n replied: {r.status_code} - {r.text[:200]}")
            except Exception as e:
                st.error(f"Webhook test failed: {e}")

st.title("🛡️ AML Name Screening")
st.caption(
    "Enter a name in **Arabic or Latin**. The app transliterates, finds the best match, "
    "decides per your existing logic, and shows **XGBoost risk** for the matched record."
)

# Prepare matcher (cached)
try:
    matcher = get_matcher(dataset_path)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Main Form ----
with st.form("query"):
    name_input = st.text_input("Full name to check", placeholder="e.g., أريج المهنا  or  arij mhana")
    submitted = st.form_submit_button("Check", type="primary")

def decide_from_match(matcher: AMLNameMatcher, match: dict):
    """
    Reproduce matcher.make_decision() logic so we can pass a custom threshold
    to find_best_match() from the UI without touching your model.py.
    """
    if not match:
        return {
            "decision": "ALLOWED",
            "confidence": 0.95,
            "reason": "No match found in the watchlist",
            "similarity": 0.0,
            "match_details": None
        }
    similarity_score = match['composite_similarity']
    risk_category = match.get('risk_category', 'Unknown')
    risk_weight = matcher.risk_weights.get(risk_category, 0.5)
    final_score = similarity_score * (0.6 + risk_weight * 0.4)

    if final_score >= 0.85 and similarity_score >= 0.9:
        decision = "BLOCKED"
        confidence = min(final_score + 0.05, 1.0)
        reason = f"Very high similarity ({similarity_score*100:.1f}%) with {risk_category}"
    elif final_score >= 0.7 and similarity_score >= 0.8:
        decision = "REVIEW"
        confidence = final_score
        reason = f"High similarity ({similarity_score*100:.1f}%) with {risk_category}"
    elif final_score >= 0.5:
        decision = "REVIEW"
        confidence = final_score * 0.8
        reason = f"Moderate similarity ({similarity_score*100:.1f}%) with {risk_category}"
    else:
        decision = "ALLOWED"
        confidence = 1 - final_score
        reason = f"Low similarity ({similarity_score*100:.1f}%)"

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "reason": reason,
        "similarity": round(similarity_score * 100, 2),
        "match_details": match
    }

if submitted:
    if not name_input.strip():
        st.warning("Please enter a valid name.")
    else:
        with st.spinner("Processing…"):
            t0 = time.time()
            # Use slider threshold for matching
            match = matcher.find_best_match(name_input, threshold=threshold)
            result = decide_from_match(matcher, match)
            dt = (time.time() - t0) * 1000

        st.write(f"⏱️ Completed in **{dt:.0f} ms**")

        # ---- Display decision ----
        decision = result["decision"]
        sim_pct = result["similarity"]
        match = result["match_details"]
        xgb_prob_pct = None

        if decision == "BLOCKED":
            st.error(f"**Decision: {decision}**")
        elif decision == "REVIEW":
            st.warning(f"**Decision: {decision}**")
        else:
            st.success(f"**Decision: {decision}**")

        cols = st.columns(3)
        cols[0].metric("Similarity", f"{sim_pct:.2f}%")
        cols[1].metric("Confidence", f"{result['confidence']*100:.1f}%")
        cols[2].metric("Reason", result["reason"])

        # ---- Match details and XGBoost risk ----
        if match:
            st.subheader("Top match")
            md_cols = st.columns(2)
            md_cols[0].write(f"**Name:** {match['full_name']}")
            md_cols[1].write(f"**Risk Category:** {match['risk_category']}")
            md_cols[0].write(f"**Nationality:** {match['nationality']}")
            md_cols[1].write(f"**Notes:** {match['notes'] if match['notes'] else '—'}")

            # XGBoost risk probability for this exact matched row
            try:
                idx = match["index"]
                X_row = matcher.df.loc[idx, matcher.feature_names].fillna(0)
                X_scaled = matcher.scaler.transform([X_row])
                xgb_prob = matcher.xgb_model.predict_proba(X_scaled)[0, 1]
                xgb_prob_pct = 100.0 * float(xgb_prob)
            except Exception:
                xgb_prob_pct = None

            if xgb_prob_pct is not None:
                st.info(f"**XGBoost risk probability:** {xgb_prob_pct:.2f}%")
            else:
                st.info("**XGBoost risk probability:** n/a")

            if show_debug:
                sims = match["similarity_scores"]
                with st.expander("Transliteration & canonical forms used"):
                    st.code(
                        f"latin_input         = {sims.get('latin_input','')}\n"
                        f"latin_candidate     = {sims.get('latin_candidate','')}\n"
                        f"canonical_input     = {sims.get('canonical_input','')}\n"
                        f"canonical_candidate = {sims.get('canonical_candidate','')}",
                        language="text"
                    )
                with st.expander("Similarity details"):
                    dbg = {k: v for k, v in sims.items()
                           if k not in ('latin_input','latin_candidate','canonical_input','canonical_candidate')}
                    st.json(dbg)
        else:
            st.info("No close match found in the watchlist.")

        # ---- Send event to n8n (webhook) - SEND ALL EVENTS ----
        try:
            event = {
                "input": name_input,
                "decision": decision,                          # ALLOWED / REVIEW / BLOCKED
                "similarity": float(sim_pct),                  # %
                "confidence": round(result["confidence"]*100, 2),  # %
                "reason": result["reason"],
                "xgbRiskPct": (None if xgb_prob_pct is None else round(xgb_prob_pct, 2)),
                "timestamp": pd.Timestamp.now().isoformat(),
                "match": None
            }
            if match:
                sims = match["similarity_scores"]
                event["match"] = {
                    "full_name": match["full_name"],
                    "risk_category": match["risk_category"],
                    "nationality": match["nationality"],
                    "notes": match["notes"],
                    "latin_input": sims.get("latin_input",""),
                    "latin_candidate": sims.get("latin_candidate",""),
                    "canonical_input": sims.get("canonical_input",""),
                    "canonical_candidate": sims.get("canonical_candidate",""),
                }
            
            # Envoyer TOUS les événements pour que n8n puisse répondre
            print(f"[DEBUG] Sending {decision} event to n8n...")
            notify_n8n(event)
                
        except Exception as e:
            print(f"[n8n] event build failed: {e}")

        # ---- Update history ----
        st.session_state.history.append({
            "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input": name_input,
            "Decision": decision,
            "Similarity %": round(sim_pct, 2),
            "Risk Category": match["risk_category"] if match else "—",
            "XGBoost Risk %": round(xgb_prob_pct, 2) if xgb_prob_pct is not None else "—",
        })

# ---- History ----
st.divider()
st.subheader("History")

filter_text = st.text_input("Filter history", placeholder="Type to filter…")
hist_df_full = format_history_df(st.session_state.history)
if filter_text.strip():
    mask = hist_df_full.apply(lambda row: row.astype(str).str.contains(filter_text, case=False, na=False)).any(axis=1)
    hist_df = hist_df_full[mask]
else:
    hist_df = hist_df_full

# 👉 Force un rendu textuel pour éviter ArrowTypeError (nombres + "—")
display_df = hist_df.copy()
for col in ["Similarity %", "XGBoost Risk %"]:
    display_df[col] = display_df[col].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else (x if x is not None else "—")
    )

st.dataframe(display_df, width="stretch")

csv = hist_df_full.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full history (CSV)",
    data=csv,
    file_name="aml_history.csv",
    mime="text/csv",
    width="stretch"
)

st.caption("Tip: adjust the dataset path in the sidebar if needed. "
           "On first run, the app will train and cache models if none are saved yet.")