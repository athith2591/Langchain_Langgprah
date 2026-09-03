from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv()

def first_chain():

    prompt = ChatPromptTemplate.from_template("You generate a marketing line based on the product name and the target audience: {product}, {target_audience}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product": "mountain bikes", "target_audience":"men"})

    print(result)

def multi_model_chain():
    """My first try 
    """

    prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
    models =["gpt-4o-mini", "gpt-4o"]


    for model in models:
        chosen_model = ChatOpenAI(model=model, temperature=0.7)
        chain = prompt | chosen_model
        response = chain.invoke({"question" : "what is AI"})
        print({model: response.content})

def multi_model_chain_2():
    """My first try 
    """

    prompt = "what is AI"
    models =["gpt-4o-mini", "gpt-4o"]


    for model_name in models:
        chosen_model = init_chat_model(
            model= model_name,
            temperature = 0.7,
            streaming = True,
            max_retries = 3
        )

        response = chosen_model.invoke(prompt)
        print({model_name: response.content})
        print()

if __name__ == "__main__":
    multi_model_chain_2()
    # first_chain()