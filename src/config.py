import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def configure_genai():
    """
    Configures the Google Generative AI SDK.
    """
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found.")
        return False
    
    genai.configure(api_key=GOOGLE_API_KEY)
    return True
