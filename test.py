from openai import OpenAI

client = OpenAI(
    api_key="apk_gwPsjitcqkprywGGYzGSJEtn0RFQ04KJWlkYwD7uHvk",
    base_url="https://api.llmadaptive.uk"
)

completion = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": "Hello! How are you today?"
        }
    ],
    max_tokens=150,
    temperature=0.7
)

print(completion.choices[0].message.content)