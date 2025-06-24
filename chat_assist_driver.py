import chat_assistant
from openai import OpenAI
from minsearch import Index, AppendableIndex
import requests 

client = OpenAI()

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

search_description = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

def add_entry(question, answer):
    doc = {
        'question': question,
        'text': answer,
        'section': 'user added',
        'course': 'data-engineering-zoomcamp'
    }
    index.append(doc)

add_entry_description = {
    "type": "function",
    "name": "add_entry",
    "description": "Add an entry to the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to be added to the FAQ database",
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}

tools = chat_assistant.Tools()
tools.add_tool(search, search_description)
tools.add_tool(add_entry, add_entry_description)
tools.get_tools()

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_interface = chat_assistant.ChatInterface()

chat = chat_assistant.ChatAssistant(
    tools=tools,
    developer_prompt=developer_prompt,
    chat_interface=chat_interface,
    client=client
)

chat.run()