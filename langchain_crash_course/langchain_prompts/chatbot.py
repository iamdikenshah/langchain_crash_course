from models import getAzureModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

def static_chat_message():
    model = getAzureModel()

    messages = [
        SystemMessage(content="You are a helpful assistemt")
    ]

    while(True):
        user_input = input('You: ')
        messages.append(HumanMessage(content=user_input))
        if user_input == 'exit':
            messages = []
            break
        
        response = model.invoke(messages)
        messages.append(AIMessage(content=response.content))
        print(messages)

def dynamic_chat_message():
    template = ChatPromptTemplate([
        ('system','You are a helpful {domain} expert'),
        ('human','Explain in simple terms, what is {topic}')])
    
    model = getAzureModel()
    chain = template | model
    result = chain.invoke({'domain': 'Cricket', 'topic': 'Guggly ball'})
    print(result.content)
     

def main():
    # static_chat_message()
    dynamic_chat_message()
        

if __name__ == '__main__':
    main()