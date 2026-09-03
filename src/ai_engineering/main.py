from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

from langchain_core import __version__ as core_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")
# print(f"langchain-core version: {core_version}")
# print(f"langchain-core version: {core_version}")

def main() -> None:

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")
    print("Set up complete!")

if __name__ == "__main__":
    main()
