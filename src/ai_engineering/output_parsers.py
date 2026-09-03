from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("write a short poem about {topic}")

llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

chain = prompt | llm | parser

response = chain.invoke({"topic":"nature"})

# print(response)
# print(type(response))

# pydantic example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    occupation: str = Field(..., description="The person's occupation")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template("Return the Json object with name, age, occupation for :{description}"
                                          ).partial(formal_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

# response = chain.invoke({"description":"Maria is a 30 year old artist"})

# print(response)

#-----------------------------------------------------------------------#
# Structured ouput

class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: str = Field(description="The rating of the movie out of 10")

structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Inception is a gripping sci-fi thriller that combines spectacular action " \
                                    "with a clever exploration of dreams and reality. Christopher Nolan’s layered " \
                                    "storytelling can be complex, but strong performances, striking visuals, and Hans Zimmer’s " \
                                    "memorable score make it a rewarding watch. 9/10 or 4.5/5 ")

print(result)