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
            "content": """
            Use information from the following research abstracts to answer the question.
            Do not use any information that is not given in the paper.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the titles of the papers you use in your answer.
            If the answer cannot be deduced from the context, state that you do not know the answer. Do not risk giving misinformation.
            Once again, it is imperative that you do not use your own knowledge or any information that is not given in the context.
            """,
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer. Remember to not use information that is not given in the context, and state the title of the paper you use as a source.

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
