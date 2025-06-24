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

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information.

Output templates:

If you want to perform search, use this template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
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

# Some of the search results will be duplicates, so we need to remove them:
def dedup(seq):
    seen = set()
    result = []
    for el in seq:
        _id = el['_id']
        if _id in seen:
            continue
        seen.add(_id)
        result.append(el)
    return result

def agentic_search(question):
    search_queries = []
    search_results = []
    previous_actions = []

    iteration = 0
    
    while True:
        print(f'ITERATION #{iteration}...')
    
        context = build_context(search_results)
        prompt = prompt_template.format(
            question=question,
            context=context,
            search_queries="\n".join(search_queries),
            previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
            max_iterations=3,
            iteration_number=iteration
        )
    
        print(prompt)
    
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        print(json.dumps(answer, indent=2))

        previous_actions.append(answer)
    
        action = answer['action']
        if action != 'SEARCH':
            break
    
        keywords = answer['keywords']
        search_queries = list(set(search_queries) | set(keywords))

        for k in keywords:
            res = search(k)
            search_results.extend(res)
    
        search_results = dedup(search_results)
        
        iteration = iteration + 1
        if iteration >= 4:
            break
    
        print()

    return answer

agentic_search('how do I prepare for the course?')