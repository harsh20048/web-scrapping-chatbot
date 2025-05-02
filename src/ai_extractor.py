import requests

def extract_with_diffbot(url, token):
    """
    Extracts structured/semantic data from a web page using the Diffbot Article API.
    Args:
        url (str): The URL of the page to extract from.
        token (str): Your Diffbot API token.
    Returns:
        dict or None: The extracted data, or None if extraction failed.
    """
    api_url = "https://api.diffbot.com/v3/article"
    params = {
        "token": token,
        "url": url
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Diffbot error: {response.status_code} {response.text}")
        return None

def extract_with_openai(text, api_key, model="gpt-3.5-turbo", prompt="Extract structured data from the following text:"):
    """
    Uses OpenAI's GPT API to extract structured/semantic data from text.
    Args:
        text (str): The text to analyze.
        api_key (str): OpenAI API key.
        model (str): Model name.
        prompt (str): Instruction prompt.
    Returns:
        dict or None: The extracted data, or None if extraction failed.
    """
    import openai
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.2,
            max_tokens=512
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"OpenAI extraction error: {e}")
        return None

# Example usage:
# result = extract_with_diffbot("https://example.com", "YOUR_DIFFBOT_TOKEN")
# print(result)
