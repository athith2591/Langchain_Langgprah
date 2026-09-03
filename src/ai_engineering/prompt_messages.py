from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")

# messages = prompt.format_messages(adjective = "funny", topic = "cats")

# print(messages)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant who translate the language {input} to {output}"),
#         ("human", "Please translate this: {text}")
#     ]
# )

# messages = prompt.format_messages(
#     input = "english",
#     output = "german",
#     text = "Hey I have seen that people think that they believe they can fly"
# )

# model = init_chat_model(
#     model= "gpt-4o-mini",
#     temperature = 0.7
# )

# repsonse = model.invoke(messages)

# print(repsonse.content)

# messages = [
#     HumanMessage(content="Hello!"),
#     AIMessage(content="Hi there! How can I assist you today?"),
#     SystemMessage(content="This is a system message"),
#     ToolMessage(content="Tool executed successfully."),
#     ChatMessage(content="This is a general chat message")
# ]

examples =[
    {"input":"happy", "output":"sad"},
    {"input":"tall", "output":"short"}
]

example_prompt=ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai","{output}")
])

fewshot_promopt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Give the opposite of each word"),
        fewshot_promopt,
        ("human", "{input}")

])

model = init_chat_model(
    model= "gpt-4o-mini",
    temperature = 0.7
)

response = model.invoke(final_prompt.format_messages(input ="stallion"))

print(response.content)


