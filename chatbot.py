import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Load environment variables if needed
load_dotenv()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="policies")

groq_api_key = os.getenv('GROQ_API_KEY', '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')
client = Groq(api_key=groq_api_key)

print("Welcome to the document chatbot! Ask questions about the documents in the database. Type 'exit' or 'quit' to end the conversation.\n")

# Conversation history for context
conversation_history = []

while True:
    user_query = input("You: ")
    if user_query.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    results = collection.query(
        query_texts=[user_query],
        n_results=4
    )

    system_prompt = f"""
You are a helpful assistant. You answer questions about documents that provide technical guidelines, license guidelines, etc. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
{results['documents']}
"""

    # Build the message list: system prompt + conversation history + new user input
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    print("\nBot:", assistant_reply, "\n")

    # Add user and assistant messages to the conversation history
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": assistant_reply})
