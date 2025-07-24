# frontend.py

import streamlit as st
import requests
import json

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000/process_claim"

# --- App Layout ---
st.set_page_config(page_title="Insurance Claim Adjudicator", layout="wide")

# Title
st.title("🤖 AI Insurance Claim Adjudicator")
st.write("Enter a claim summary in plain English. The AI will use the policy documents to make a decision.")

# User Input
user_query = st.text_area("Enter your claim details here:", "A 46-year-old male had knee surgery in Pune. His policy is 3 months old.", height=100)

# Submit button
if st.button("Adjudicate Claim"):
    if user_query:
        # Show a spinner while processing
        with st.spinner("Analyzing claim against policy documents..."):
            try:
                # Call the FastAPI backend
                payload = {"query": user_query}
                response = requests.post(FASTAPI_URL, json=payload)

                if response.status_code == 200:
                    # Display the results
                    result = response.json()
                    st.divider()
                    st.subheader("Decision Result:")
                    
                    decision = result.get("decision", "N/A")
                    if decision == "Approved":
                        st.success(f"**Decision: {decision}**")
                    else:
                        st.error(f"**Decision: {decision}**")
                    
                    st.info(f"**Justification:** {result.get('justification', 'No justification provided.')}")
                    st.warning(f"**Approved Amount:** ${result.get('amount', 0):,}")

                    # Expander for the raw JSON response
                    with st.expander("Show Raw JSON Response"):
                        st.json(result)
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend API. Please ensure it's running. Error: {e}")
    else:
        st.warning("Please enter claim details before adjudicating.")