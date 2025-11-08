import os
from openai import OpenAI

def ai_summarize(text, model="nvidia/nemotron-nano-12b-v2-vl:free"):
    """
    Summarize long text using an AI model from OpenRouter.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("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")
    )

    # If text is too long, trim it (most APIs have input limits)
    text = text[:8000]

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an academic paper summarizer. Summarize the core ideas clearly and concisely."},
            {"role": "user", "content": text}
        ]
    )

    return completion.choices[0].message.content
