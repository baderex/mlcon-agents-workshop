import requests 
from minsearch import Index, AppendableIndex
from openai import OpenAI
import json

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

client = OpenAI()

prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT> 
{context}
</CONTEXT>

If CONTEXT is EMPTY, you can use our FAQ database.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}
""".strip()

def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


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

def build_context(search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    return context.strip()

def agentic_rag_v1(question):
    context = "EMPTY"
    prompt = prompt_template.format(question=question, context=context)
    answer_json = llm(prompt)
    answer = json.loads(answer_json)
    print(answer)

    if answer['action'] == 'SEARCH':
        print('need to perform search...')
        search_results = search(question)
        context = build_context(search_results)
        
        prompt = prompt_template.format(question=question, context=context)
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        print(answer)

    return answer

agentic_rag_v1('how do I join the course?')
agentic_rag_v1('how patch KDE under FreeBSD?')