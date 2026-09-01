import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads your .env file

client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant for a legal firm. Answer only questions related to legal topics.",
    messages=[
        {"role": "user", "content": "What is defined as self defense?"},
        {"role": "assistant", "content": "Self-defense is a legal defense that allows a person to use reasonable force to protect themselves from imminent harm or threat of harm. The force used must be proportionate to the threat faced, and the person claiming self-defense must not have provoked the situation. The specific laws regarding self-defense can vary by jurisdiction."},
        {"role": "user", "content": "If someone slaps me in the face, can I hit them back?"}
    ]
)

print(message.content[0].text)