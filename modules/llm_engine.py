import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def detect_question(text):

    prompt = f"""
You are a question detection system.

Analyze the following conversation text.

Determine whether the speaker is asking a question.

If it is a question, extract the complete question.
If it is not a question, return exactly: NO_QUESTION

Conversation:
{text}

Return only the question or NO_QUESTION.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    if result == "NO_QUESTION":
        return None

    return result


def generate_answer(question):

    payload = {
        "model": MODEL_NAME,
        "prompt": (
            "Answer this question clearly and concisely: "
            + question
        ),
        "stream": True
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(10, 180)
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if line:

            data = json.loads(line)

            token = data.get("response", "")

            if token:
                yield token