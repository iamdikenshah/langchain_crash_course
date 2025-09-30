from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub

load_dotenv()


def get_llm():
    llm = OpenAI(temperature=0.7, 
                 model="gpt-3.5-turbo-instruct", 
                 max_tokens=150)
    return llm

def create_agent():
    agent = initialize_agent(llm=get_llm(),
                             tools=get_tools(),
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    return agent

def create_agent_with_executor():
    """Create agent using AgentExecutor with more control"""
    # Get the react prompt from hub
    prompt = hub.pull("hwchase17/react")
    
    # Create the react agent
    agent = create_react_agent(get_llm(), get_tools(), prompt)
    
    # Create AgentExecutor with custom settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=get_tools(),
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent_executor

def get_tools():
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    duckduckgo_search = DuckDuckGoSearchRun()
    return [wikipedia, duckduckgo_search]

def main():
    print("Main called")
    query = input("Enter your query: ")
    
    print("\n=== Using Standard Agent ===")
    agent = create_agent()
    output = agent.invoke(query)
    print(f"Standard Agent Output: {output}")
    
    print("\n=== Using AgentExecutor ===")
    agent_executor = create_agent_with_executor()
    output_executor = agent_executor.invoke({"input": query})
    print(f"AgentExecutor Output: {output_executor}")

if __name__ == '__main__':
    main()