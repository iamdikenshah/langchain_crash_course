from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

def main():
    print("Main Called")


def configure_embedding_model() -> OpenAIEmbeddings:
    embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
    return embedding

def generate_documents():
    documents = ["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership."
                "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
                "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records."
                "Rohit Sharma is known for his elegant batting and record-breaking double centuries."
                "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]
    return documents

if __name__ == '__main__':
    main()