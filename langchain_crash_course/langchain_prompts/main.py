from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

if __name__ == "__main__":
    llm = AzureChatOpenAI(deployment_name="gpt-4o", 
                          temperature=1.7, 
                          api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
    
    response = llm.invoke("Write a five line poem about the sea.")
    print(response.content)
