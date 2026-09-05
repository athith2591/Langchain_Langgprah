# Section 1 Project: Smart Q & A Bot. A production ready question-answering bot with structured output.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langsmith import traceable, Client
import os

load_dotenv()

# Langsmith configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT_NAME", "Smart Q&A Bot Project")
    # os.environ["LANGSMITH_PROJECT_NAME"] = "Smart Q&A Bot Project"
    print(f"Langsmith is configured. -Project:{os.getenv('LANGSMITH_PROJECT_NAME')}")


class QAResponse(BaseModel):
    answer: str =Field(description="The answers to the users questions.")
    confidence: str =Field(description="Confidence level: high, medium or low")
    reasoning: str =Field(description="The reason behind the answer provided")
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="A list of follow-up questions related to the topic."
    )
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer.", default=False
    )

class SmartQABot:
    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 ):
        self.model = ChatOpenAI(model=model_name, 
                                temperature=temperature).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
        """You are a knowledgeable assistant
        Your guidelines:
        - Answer questions accurately and concisely
        - Be honest about uncertainity - set confidence to 'low' if unsure
        - Provide clear reasoning for your answers
        - Suggest relevant follow up questions.
        - Indicate if external sources would help

        Always respond with accurate helpful information.
        """
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )
        self.chain = self.prompt | self.model 

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question:str) -> QAResponse:
        try:
            response = self.chain.invoke({"question":question})
            return response
        except Exception as e:
            return QAResponse(
                answer = "I am sorry, I couldn't answer this question at this time",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=["please try agian later"],
                sources_needed=True
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str])-> List[QAResponse]:
        """Ask multiple questions in parallel
        """
        inputs =[{"question":q} for q in questions]
        return self.chain.batch(inputs)


def demo_qa_bot():
    bot = SmartQABot()

    questions =[
        "What is the capital of France",
        "Explain the theory of relativity",
        "How does photosynthesis work"
    ]

    print("=" * 60)
    print("Smart Q&A bot demo")
    print("=" * 60)

    for question in questions:
         print(f"\n Question: {question}")
         print("_" * 40)
         response = bot.ask(question)

         print(f"Question: {question}")
         print(f"Answer: {response.answer}")
         print(f"Confidence: {response.confidence}")
         print(f"Reasoning: {response.reasoning}")
         print(f"Follow-up Questions: {response.follow_up_questions}")
         print(f"Sources Needed: {response.sources_needed}")
         print("-" * 60)

@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
    bot = SmartQABot()
    print("Error handling demo")
    print("=" * 60)

    # Testing with a very long question (edge case)

    long_question = "What is " + "very " *100 + "important?"

    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")

@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
    bot = SmartQABot()

    questions =[
        "What Python?",
        "what is Rust?",
        "What is JavaScript?"
    ]

    print("=" * 60)
    print("Smart Q&A bot batch demo")
    print("=" * 60)

    responses = bot.ask_batch(questions)
    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f" ->{r.answer[:100]}...")
        print(f" Confidence: {r.confidence}")

client = Client()

if __name__ =="__main__":

    try:
        demo_qa_bot()
        demo_batch_processing()
        demo_error_handling()
    finally:
        client.flush() # Ensures all traces are sent to Langsmith


                                                                     

