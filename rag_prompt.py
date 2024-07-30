from huggingface_hub import InferenceClient

from vector_search import generate_context

token = open("api_token.txt").read()

client = InferenceClient(
    "HuggingFaceH4/zephyr-7b-beta",
    token=token
)

def prompt(query):
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Use information from only one of the given movie plots to answer the question. Do not use any outside or other information.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the title of the source movie, then provide the answer.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    
    context = generate_context(query)
    final_prompt = prompt_in_chat_format
    final_prompt[1]["content"] = final_prompt[1]["content"].format(question = query, context=context)

    ans = ""
    for message in client.chat_completion(
	messages=final_prompt,
	max_tokens=500,
	stream=True,
):
        ans += message.choices[0].delta.content
    
    return ans

# prompt("who is the world's most super-bad turned super-dad?")
# prompt("Who do earth's mightiest heroes have to fight against?")
# prompt("who is part of the rebellion against reality's controller robots?")
# prompt("what was Po the Panda chosen as?")
# prompt("where does Po the Panda live?")
# prompt("who prevents cinderella from going to the ball?")
# prompt("who does ariel the mermaid princess want to meet?")