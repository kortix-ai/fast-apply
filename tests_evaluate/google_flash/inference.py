# pip install google-generativeai

import os
import google.generativeai as genai

os.environ["GEMINI_KEY"] = os.getenv("GEMINI_KEY")


genai.configure(api_key=os.environ["GEMINI_KEY"])

prompt = "Tell me a joke"

generation_config = {
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}


model = genai.GenerativeModel(
  #  model_name="gemini-1.5-flash-latest",
  model_name="tunedModels/train-4gaullhp8hak",
  generation_config=generation_config,
  #  system_instruction=system_prompt,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message(prompt)

print(response.text)
