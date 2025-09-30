from dotenv import load_dotenv
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint

load_dotenv()

def main():
    # llm = OpenAI(model='gpt-3.5-turbo-instruct')
    # result = llm.invoke("What is the capital of India")
    # print(result)

    # openAI_chatModels()
    huggingFace_model()  # Commented out until API key is configured
    

def openAI_chatModels():
    model = ChatOpenAI(model='gpt-4o-mini', 
                       temperature=0.7)
    result = model.invoke("What is the capital city of India?")
    print(result.content)

def antropic_ChatModel():
    llm = ChatAnthropic(model_name="", temperature=0.7)
    result = llm.invoke("What is the capital city of India?")
    print(result.content)

def gemini_ChatModel():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    result = llm.invoke("What is the capital city of India?")
    print(result.content)

def huggingFace_model():
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        huggingfacehub_api_token=api_key
    )
    model = ChatHuggingFace(llm=llm)
    result = model.invoke("What is the prime minister of USA?")
    print(result.content)


if __name__ == '__main__':
    main()