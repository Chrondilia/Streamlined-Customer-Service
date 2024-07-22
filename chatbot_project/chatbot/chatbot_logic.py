# main.py

import openai
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv('sk-proj-30EoPOxtdY1wnmOv6QMtT3BlbkFJiumXoZFSoYVAADoCtO2v')

def chatbot_response(user_message):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Use the appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Example of usage (optional for testing)
if __name__ == "__main__":
    user_message = "Hello, how are you?"
    print(chatbot_response(user_message))
