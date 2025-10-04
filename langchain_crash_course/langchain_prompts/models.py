from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def getAzureModel() -> AzureChatOpenAI:
    model = AzureChatOpenAI(deployment_name="gpt-4o-mini", 
                          temperature=0, 
                          api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                          api_key=os.getenv("AZURE_OPENAI_API_KEY"))
    return model