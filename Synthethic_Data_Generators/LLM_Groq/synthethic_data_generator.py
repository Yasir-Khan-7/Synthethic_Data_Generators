import re
import pandas as pd
import os
from io import StringIO
from groq import Groq

# Set API key
GROQ_API_KEY = "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"

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# **🔹 Set your desired prompt here**
USER_PROMPT = input("please enter your prompt:\n")

def extract_table_from_response(response_text):
    """Extracts table data from CSV format."""
    
    # If response contains `"` or `,`, assume it is CSV
    if '"' in response_text or "," in response_text:
        try:
            df = pd.read_csv(StringIO(response_text))
            return df
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return None
    else:
        print("❌ No valid table data found.")
        return None

def generate_synthetic_data():
    """Prompt the LLM and generate a CSV file as 'synthethic_data.csv'."""
    
    csv_filename = "synthethic_data.csv"

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": USER_PROMPT}],
            model="llama-3.3-70b-versatile"
        )

        raw_output = chat_completion.choices[0].message.content
        print(f"\n✅ LLM Response:\n{raw_output}\n")

        df = extract_table_from_response(raw_output)

        if df is not None and not df.empty:
            df.to_csv(csv_filename, index=False)
            print(f"✅ Data successfully saved as '{csv_filename}'")

    except Exception as e:
        print(f"❌ Groq API Error: {e}")

if __name__ == "__main__":
    generate_synthetic_data()
