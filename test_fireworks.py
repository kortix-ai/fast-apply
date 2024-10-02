import requests
import json

url = "https://api.fireworks.ai/inference/v1/completions"
payload = {
    "model": "accounts/marko-1d84ff/models/1b",
    "prompt": "The sky is",
    "max_tokens": 4096,
    #  "echo": True,
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "stream": True,
    "context_length_exceeded_behavior": "truncate",
}
headers = {
    "Authorization": "Bearer fw_3ZhfovPCeNKHpHcEnS9D9HmX",
    "Content-Type": "application/json"
}

def stream_response():
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        json_data = json.loads(json_str)
                        text = json_data['choices'][0]['text']
                        print(text, end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {json_str}")
    print()  # Print a newline at the end

stream_response()
