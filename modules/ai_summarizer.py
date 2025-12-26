# modules/ai_summarizer.py

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Defaults
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.3


def _prompt_for_refinement(themed_input: dict):
    """
    Build the prompt messages for abstractive summarization.
    """
    system = (
        "You are an expert academic summarization engine. "
        "Synthesize the provided thematically grouped excerpts "
        "into a single, cohesive academic abstract."
    )

    input_text = ""
    for theme, text in themed_input.items():
        input_text += f"[{theme}]\n{text}\n\n"

    user = (
        "Create a concise academic abstract (150–200 words) from the text below. "
        "Use formal academic tone, smooth transitions, and logical flow. "
        "Do not add new information.\n\n"
        f"{input_text}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def refine_with_openai(
    themed_input: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    retry: int = 2,
):
    """
    Refine extractive summary into an abstractive summary using OpenAI.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    messages = _prompt_for_refinement(themed_input)

    last_exc = None
    for attempt in range(retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)

    raise RuntimeError(f"OpenAI API failed: {last_exc}")
