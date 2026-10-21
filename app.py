import json
import os
from typing import TypedDict
from google import genai
from langgraph.graph import END, START, StateGraph
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Hybrid AI Retention Agent", page_icon="🚀", layout="wide"
)


# ==========================================
# 1. CACHED ML MODEL & RAG ENGINE SETUP
# ==========================================
@st.cache_resource
def initialize_ai_engines():
  # A. Train Random Forest Classifier on Synthetic Telemetry
  np.random.seed(42)
  n_samples = 1000
  data = {
      "days_since_last_login": np.random.randint(1, 90, n_samples),
      "support_tickets_opened": np.random.randint(0, 8, n_samples),
      "monthly_spend_usd": np.round(
          np.random.uniform(10.0, 1000.0, n_samples), 2
      ),
      "cart_abandonment_rate": np.round(
          np.random.uniform(0.0, 1.0, n_samples), 2
      ),
  }
  df = pd.DataFrame(data)

  churn_prob = (
      (df["days_since_last_login"] / 90) * 0.4
      + (df["support_tickets_opened"] / 8) * 0.4
      + (df["cart_abandonment_rate"]) * 0.2
      - (df["monthly_spend_usd"] / 2000)
  )
  df["churned"] = (churn_prob > 0.42).astype(int)

  X = df[[
      "days_since_last_login",
      "support_tickets_opened",
      "monthly_spend_usd",
      "cart_abandonment_rate",
  ]]
  y = df["churned"]

  X_train, _, y_train, _ = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
  ml_model.fit(X_train, y_train)

  # B. Initialize In-Memory Vector Store for RAG Playbooks
  playbooks = [
      "Policy VIPHighSpend: If a customer spends over $400/month and has a churn risk above 0.70, offer a 30% VIP Discount Code (VIP30) and free priority shipping for 6 months.",
      "Policy SupportFrustration: If a customer has opened 3 or more support tickets and is at risk of churning, apologize for the service delays, escalate their ticket to Tier-2 Engineering, and issue a $25 store credit.",
      "Policy PriceSensitive: If a customer has a high cart abandonment rate (> 0.60) and spends under $400/month, offer a 15% discount code (SAVE15) valid for 48 hours to help clear their cart.",
      "Policy StandardCheckIn: For moderate churn risk between 0.50 and 0.70 with low support tickets, send a friendly product recommendation newsletter with no monetary discount.",
  ]
  vectorizer = TfidfVectorizer()
  doc_vectors = vectorizer.fit_transform(playbooks)

  return ml_model, vectorizer, doc_vectors, playbooks


ml_model, vectorizer, doc_vectors, playbooks = initialize_ai_engines()


def predict_customer_churn(
    days_inactive: int, tickets: int, spend: float, cart_abandon: float
) -> float:
  features = pd.DataFrame(
      [[days_inactive, tickets, spend, cart_abandon]],
      columns=[
          "days_since_last_login",
          "support_tickets_opened",
          "monthly_spend_usd",
          "cart_abandonment_rate",
      ],
  )
  prob = ml_model.predict_proba(features)[0][1]
  return round(float(prob), 3)


def retrieve_retention_policy(query: str) -> str:
  query_vec = vectorizer.transform([query])
  similarities = cosine_similarity(query_vec, doc_vectors).flatten()
  top_indices = similarities.argsort()[-2:][::-1]
  return "\n\n".join([playbooks[i] for i in top_indices])


# ==========================================
# 2. LANGGRAPH AUTONOMOUS WORKFLOW
# ==========================================
class RetentionActionPlan(BaseModel):
  risk_category: str = Field(
      description="Low, Medium, or Critical churn risk"
  )
  applied_policy: str = Field(
      description="Name of the company policy selected from RAG"
  )
  discount_or_credit_offered: str = Field(
      description="Exact coupon code or credit amount granted"
  )
  personalized_email_draft: str = Field(
      description="Ready-to-send personalized retention email to the customer"
  )


class ChurnAgentState(TypedDict):
  customer_id: str
  customer_name: str
  days_inactive: int
  support_tickets: int
  monthly_spend: float
  cart_abandonment: float
  recent_complaint_note: str
  ml_churn_score: float
  retrieved_policies: str
  final_action_plan: dict


def ml_scoring_node(state: ChurnAgentState) -> dict:
  score = predict_customer_churn(
      state["days_inactive"],
      state["support_tickets"],
      state["monthly_spend"],
      state["cart_abandonment"],
  )
  return {"ml_churn_score": score}


def rag_retrieval_node(state: ChurnAgentState) -> dict:
  search_query = (
      f"VIP spend ${state['monthly_spend']} month, opened {state['support_tickets']} "
      f"support tickets delay frustration, cart abandonment rate {state['cart_abandonment']} price sensitive, "
      f"churn risk {state['ml_churn_score']}. {state['recent_complaint_note']}"
  )
  policies = retrieve_retention_policy(search_query)
  return {"retrieved_policies": policies}


def llm_decision_agent_node(state: ChurnAgentState) -> dict:
  api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
  if not api_key:
    return {
        "final_action_plan": {
            "risk_category": "Configuration Error",
            "applied_policy": "Missing GEMINI_API_KEY",
            "discount_or_credit_offered": "N/A",
            "personalized_email_draft": (
                "Please add GEMINI_API_KEY in Streamlit Cloud App Settings -> Secrets."
            ),
        }
    }

  ai_client = genai.Client(api_key=api_key)

  prompt = f"""
    You are an Autonomous Customer Retention AI Agent.
    Analyze the customer telemetry, their ML Churn Score, their typed message, and the retrieved Company Playbooks.
    Select the exact matching policy and write a warm, empathetic, personalized win-back email that directly addresses what they typed in their feedback note.

    Customer Profile:
    - Name: {state['customer_name']} (ID: {state['customer_id']})
    - Days Since Last Login: {state['days_inactive']}
    - Support Tickets Opened: {state['support_tickets']}
    - Monthly Spend: ${state['monthly_spend']}
    - Cart Abandonment Rate: {state['cart_abandonment']}
    - Customer's Typed Message/Feedback: "{state['recent_complaint_note']}"
    - ML Churn Probability: {state['ml_churn_score']}

    Retrieved Company Retention Playbooks (Follow strictly):
    {state['retrieved_policies']}
    """

  response = ai_client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={
          "response_mime_type": "application/json",
          "response_schema": RetentionActionPlan,
          "temperature": 0.2,
      },
  )

  return {"final_action_plan": json.loads(response.text)}


def low_risk_skip_node(state: ChurnAgentState) -> dict:
  return {
      "retrieved_policies": (
          "Skipped RAG & LLM (Customer predicted as Low Risk < 50% by ML Gatekeeper to save API compute costs)."
      ),
      "final_action_plan": {
          "risk_category": "Low (Healthy Customer)",
          "applied_policy": "None Required (Loyal Customer)",
          "discount_or_credit_offered": "None",
          "personalized_email_draft": (
              f"Hi {state['customer_name']}, thank you for being an active customer and sharing your feedback: '{state['recent_complaint_note']}'. No retention intervention needed!"
          ),
      },
  }


def route_by_churn_risk(state: ChurnAgentState) -> str:
  if state["ml_churn_score"] >= 0.50:
    return "trigger_rag_intervention"
  return "mark_safe"


@st.cache_resource
def get_compiled_graph():
  workflow = StateGraph(ChurnAgentState)
  workflow.add_node("ml_scorer", ml_scoring_node)
  workflow.add_node("rag_retriever", rag_retrieval_node)
  workflow.add_node("decision_agent", llm_decision_agent_node)
  workflow.add_node("safe_skip", low_risk_skip_node)

  workflow.add_edge(START, "ml_scorer")
  workflow.add_conditional_edges(
      "ml_scorer",
      route_by_churn_risk,
      {"trigger_rag_intervention": "rag_retriever", "mark_safe": "safe_skip"},
  )
  workflow.add_edge("rag_retriever", "decision_agent")
  workflow.add_edge("decision_agent", END)
  workflow.add_edge("safe_skip", END)
  return workflow.compile()


retention_app = get_compiled_graph()

# ==========================================
# 3. STREAMLIT INTERACTIVE WEB UI
# ==========================================
st.title(
    "🚀 Autonomous Customer Churn & Retention Agent (ML + RAG + LangGraph)"
)
st.markdown("""
**Architecture Overview:**
1. **Predictive ML Gatekeeper:** A `RandomForestClassifier` trained on customer telemetry predicts real-time churn probability.
2. **Conditional Cost Routing:** If churn risk is $< 50\%$, `LangGraph` skips LLM & Vector DB calls to save API costs.
3. **Agentic RAG Intervention:** If risk is $\ge 50\%$, our Vector Engine retrieves company retention rules and **Gemini 2.5 Flash** generates a structured JSON retention action and custom response addressing the customer's exact typed message.
""")

col1, col2 = st.columns([1, 1.2])

with col1:
  st.subheader("1. Customer Telemetry & Live Message")
  customer_name = st.text_input("Customer Name", value="Rohan Verma")
  user_message = st.text_area(
      "Type Customer Feedback / Support Complaint",
      value=(
          "I tried using my coupon twice and the checkout crashed! Plus"
          " delivery takes way too long now."
      ),
      height=100,
  )
  days_inactive = st.slider(
      "Days Since Last Login", min_value=1, max_value=90, value=55
  )
  support_tickets = st.slider(
      "Support Tickets Opened", min_value=0, max_value=10, value=4
  )
  monthly_spend = st.slider(
      "Monthly Spend ($ USD)", min_value=10, max_value=1000, value=520, step=10
  )
  cart_abandonment = st.slider(
      "Cart Abandonment Rate", min_value=0.0, max_value=1.0, value=0.75, step=0.05
  )

  run_button = st.button(
      "⚡ Run Autonomous AI Pipeline", type="primary", use_container_width=True
  )

with col2:
  st.subheader("2. Live Pipeline Diagnostics (ML + RAG)")
  if run_button:
    with st.spinner("Running ML Gatekeeper, RAG Retrieval & LangGraph Agent..."):
      input_state = {
          "customer_id": "LIVE-WEB-USER",
          "customer_name": customer_name,
          "days_inactive": days_inactive,
          "support_tickets": support_tickets,
          "monthly_spend": float(monthly_spend),
          "cart_abandonment": float(cart_abandonment),
          "recent_complaint_note": user_message,
      }
      result = retention_app.invoke(input_state)
      plan = result["final_action_plan"]
      churn_prob = f"{result['ml_churn_score'] * 100:.1f}%"

      m1, m2 = st.columns(2)
      m1.metric("Scikit-Learn ML Churn Risk", churn_prob)
      m2.metric("Risk Tier Classification", plan["risk_category"])

      st.info(f"**Selected Company Playbook:** {plan['applied_policy']}")
      st.success(
          f"**Autonomous Reward Triggered:** {plan['discount_or_credit_offered']}"
      )

      with st.expander("🔍 View Retrieved RAG Vector Rules", expanded=True):
        st.write(result.get("retrieved_policies", "Skipped RAG."))

      st.subheader("3. Generated Personalized Response")
      st.text_area(
          "Auto-Drafted Customer Email",
          value=plan["personalized_email_draft"],
          height=200,
      )
  else:
    st.info(
        "👈 Adjust the sliders, type a custom message on the left, and click"
        " **Run Autonomous AI Pipeline** to test the system!"
    )
