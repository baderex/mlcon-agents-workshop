import chat_assistant

tools = chat_assistant.Tools()
tools.add_tool(search, search_description)

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