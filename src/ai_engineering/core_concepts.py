from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"question":"How easy is it to get a junior Ai enginner postion in Accenture"})

    print(f"Response: {result}")

    return chain

def demo_batch_execution():

    prompt = ChatPromptTemplate.from_template("Translate to German: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [{"text": "What is the fastest you ever driven?"},
           {"text": "I can fly in the sky"},
           {"text":"How long is you nose??"}
           ]

    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"inputs:{text[0]['text']} => output:{text[1]}")

def demo_streaming():
    prompt = ChatPromptTemplate.from_template("Write haiku about:{topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("Streaming output: ")
    # print(chain.stream({"topic": "nature"}))
    for chunk in chain.stream({"topic":"nature"}):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    demo_streaming()