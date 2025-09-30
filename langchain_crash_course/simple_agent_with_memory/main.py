"""
Simple Conversational Chain with Memory

This application demonstrates how to create a conversational AI that remembers
the last 5 interactions using LangChain's ConversationBufferWindowMemory.
The bot maintains context for recent exchanges while forgetting older conversations.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

load_dotenv()


def create_llm():
    """
    Create and configure the OpenAI Language Model instance.
    
    Returns:
        OpenAI: Configured LLM with moderate creativity
    """
    llm = OpenAI(temperature=0.7, max_tokens=200)
    return llm


def create_memory():
    """
    Create conversation memory to maintain chat history for last 5 conversations.
    
    Returns:
        ConversationBufferWindowMemory: Memory instance that stores only the last 5 conversation exchanges
    """
    memory = ConversationBufferWindowMemory(k=5)
    return memory


def create_conversation_prompt():
    """
    Create a custom prompt template for the conversation.
    
    Returns:
        PromptTemplate: Template that includes conversation history
    """
    template = """You are a helpful and friendly AI assistant. You remember our previous conversation.

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    return prompt


def create_conversation_chain():
    """
    Create the main conversation chain with memory.
    
    Returns:
        ConversationChain: Chain that can maintain conversational context
    """
    llm = create_llm()
    memory = create_memory()
    prompt = create_conversation_prompt()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True  # Shows the conversation flow
    )
    
    return conversation


def main():
    """
    Main conversation loop that allows continuous chat with memory.
    """
    print("=== Simple Conversational AI with Memory ===")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)
    
    # Initialize the conversation chain
    conversation = create_conversation_chain()
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nAI: Goodbye! Have a great day!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Get AI response using the conversation chain
            response = conversation.predict(input=user_input)
            print(f"\nAI: {response}")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == '__main__':
    main()