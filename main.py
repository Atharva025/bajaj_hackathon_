# main.py

import json
from fastapi import FastAPI
from pydantic import BaseModel
from query_engine import get_response # Import our core logic

# Initialize the FastAPI app
app = FastAPI(
    title="Insurance Claim Adjudicator API",
    description="An API to process natural language insurance queries against policy documents.",
    version="1.0.0"
)

# Define the request body structure using Pydantic
class QueryRequest(BaseModel):
    query: str

# Define the API endpoint for processing claims
@app.post("/process_claim")
def process_claim(request: QueryRequest):
    """
    Accepts a natural language query and returns a structured decision
    regarding an insurance claim.
    """
    # Get the string response from the RAG chain
    response_str = get_response(request.query)

    # --- Start: Robust JSON Parsing Logic ---
    # The LLM might wrap its JSON output in Markdown code blocks (e.g., ```json...```).
    # We need to strip these wrappers before parsing the string as JSON.
    
    cleaned_response_str = response_str.strip() # Remove any leading/trailing whitespace

    # Check and remove the starting Markdown fence
    if cleaned_response_str.startswith("```json"):
        cleaned_response_str = cleaned_response_str[len("```json"):].strip()
    # Check and remove the ending Markdown fence
    if cleaned_response_str.endswith("```"):
        cleaned_response_str = cleaned_response_str[:-len("```")].strip()
    # --- End: Robust JSON Parsing Logic ---

    # Attempt to convert the (now cleaned) JSON string to a Python dictionary
    try:
        response_json = json.loads(cleaned_response_str)
        return response_json
    except json.JSONDecodeError:
        # If parsing still fails, it means the LLM's core output wasn't valid JSON,
        # or there was an unexpected formatting.
        return {
            "error": "Failed to parse LLM response as JSON",
            "message": "The LLM's output could not be converted into a valid JSON object.",
            "raw_llm_response": response_str, # Original, uncleaned response for debugging
            "attempted_parse_string": cleaned_response_str # The string we tried to parse
        }

# Define a root endpoint for basic API information
@app.get("/")
def read_root():
    """
    Provides a welcome message and basic instructions for using the API.
    """
    return {"message": "Welcome to the Claim Adjudicator API. Use the /process_claim endpoint to make a query."}