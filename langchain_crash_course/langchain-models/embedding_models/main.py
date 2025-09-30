from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()

def main():
    print("Main Called")
    embedded_document()

def embedded_single_query():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
    result = embedding.embed_query("What is the capital city of the India?")
    print(str(result))

def embedded_document():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
    documents = ["Delhi is the capital of India",
                 "Gandhinagar is the capital of the Gujarat",
                 "Mumbai is a Financial capital of the India"]
    result = embedding.embed_documents(documents)
    print(str(result))

if __name__ == '__main__':
    main()