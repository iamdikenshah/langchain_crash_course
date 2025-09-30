"""
LangChain Restaurant Name and Menu Generator

This application demonstrates two different approaches to chaining LLM operations:
1. Manual chaining: Executing chains individually and passing outputs manually
2. Sequential chaining: Using LangChain's SequentialChain for automatic data flow

The app generates restaurant names based on cuisine type and then creates 
vegetarian menu suggestions for the generated restaurant name.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load environment variables from .env file (contains OpenAI API key)
load_dotenv()

def create_llm() -> OpenAI:
    """
    Create and configure the OpenAI Language Model instance.
    
    Returns:
        OpenAI: Configured LLM with moderate creativity (temperature=0.7)
                Temperature controls randomness: 0=deterministic, 1=very creative
    """
    llm = OpenAI(temperature=0.7)
    return llm

def create_restaurant_name_prompt_template():
    """
    Create a prompt template for generating restaurant names based on cuisine type.
    
    This template takes a cuisine type as input and asks the LLM to suggest
    a fancy restaurant name. The prompt is designed to return only a single name
    to ensure consistent output format.
    
    Returns:
        PromptTemplate: Template with 'cuisine' as input variable
    """
    prompt_template = PromptTemplate(
        input_variables=['cuisine'],
        template= "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. Only single name"
    )
    return prompt_template

def create_restaurant_menu_prompt_template():
    """
    Create a prompt template for generating vegetarian menu items for a restaurant.
    
    This template takes a restaurant name as input and generates vegetarian
    menu suggestions. The output is requested as a comma-separated list
    for easy parsing and display.
    
    Returns:
        PromptTemplate: Template with 'restaurant_name' as input variable
    """
    prompt_template = PromptTemplate(
        input_variables=['restaurant_name'],
        template= "Suggest some veg menu items for {restaurant_name}. Return it as comma separated list."
    )
    return prompt_template

def main_with_sequential_chain(cuisine: str, 
                               llm: OpenAI = None, 
                               restaurant_name_prompt: PromptTemplate = None, restaurant_menu_prompt: PromptTemplate = None):
    """
    Demonstrate LangChain's SequentialChain for automatic data flow between chains.
    
    This approach uses SequentialChain to automatically pass the output of the first
    chain (restaurant name) as input to the second chain (menu generation).
    Benefits:
    - Automatic data flow management
    - Built-in verbose logging
    - Clean separation of chain logic
    - Easy to extend with more chains
    
    Args:
        llm: The language model instance
        restaurant_name_prompt: Template for generating restaurant names
        restaurant_menu_prompt: Template for generating menu items
        
    Returns:
        dict: Contains both 'restaurant_name' and 'menu_items' keys
    """
    print("Sequential Chain Implementation Called")
        
    # Create individual LLMChains with specific output keys
    # Output keys are crucial for SequentialChain to route data correctly
    name_chain = LLMChain(
        llm=llm,
        prompt=restaurant_name_prompt,
        output_key="restaurant_name"  # This output becomes input for the next chain
    )
    
    menu_chain = LLMChain(
        llm=llm,
        prompt=restaurant_menu_prompt,
        output_key="menu_items"  # Final output key
    )
    
    # Create SequentialChain that automatically manages data flow
    # The first chain's output_key must match the second chain's input variable
    sequential_chain = SequentialChain(
        chains=[name_chain, menu_chain],
        input_variables=["cuisine"],  # Initial input to the chain sequence
        output_variables=["restaurant_name", "menu_items"],  # Final outputs to return
        verbose=True  # Enables detailed logging of each step
    )
    
    # Execute the entire chain sequence with a single invoke call
    result = sequential_chain.invoke({"cuisine": cuisine})
    
    # Display results on console
    print(f"\nSequential Chain Final Output: {result}")
    print(f"Restaurant: {result['restaurant_name']}")
    print(f"Menu: {result['menu_items']}")
    
    # Save results to a text file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"restaurant_output_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("=== LangChain Restaurant Generator Output ===\n")
            file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Cuisine Type: {cuisine}\n")
            file.write("-" * 50 + "\n\n")
            file.write(f"Sequential Chain Final Output: {result}\n\n")
            file.write(f"Restaurant Name: {result['restaurant_name']}\n\n")
            file.write(f"Menu Items: {result['menu_items']}\n")
            file.write("\n" + "=" * 50)
        
        print(f"\n✅ Results saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving to file: {e}")
    
    return result

def main(cuisine: str,
         llm: OpenAI = None,
         restaurant_name_prompt: PromptTemplate = None,
         restaurant_menu_prompt: PromptTemplate = None):
    """
    Demonstrate manual chaining approach using LangChain's pipe operator.
    
    This approach manually manages the data flow between chains, giving more
    control over the process but requiring explicit handling of intermediate results.
    Benefits:
    - Full control over data flow
    - Easy to add custom logic between steps
    - Clear understanding of each step
    - Simpler debugging
    
    Args:
        llm: The language model instance
        restaurant_name_prompt: Template for generating restaurant names
        restaurant_menu_prompt: Template for generating menu items
    """
    print("Main Called")
    
    
    # Create individual chains using the pipe operator (|)
    # This creates a runnable chain: prompt_template | llm
    name_chain = restaurant_name_prompt | llm
    menu_chain = restaurant_menu_prompt | llm
    
    # Execute chains manually in sequence
    # Step 1: Generate restaurant name based on cuisine
    restaurant_name_output = name_chain.invoke({"cuisine": cuisine})
    
    # Step 2: Generate menu items using the restaurant name from step 1
    menu_output = menu_chain.invoke({"restaurant_name": restaurant_name_output})
    
    # Display results on console
    print(f"\nFinal Output:")
    print(f"Restaurant: {restaurant_name_output}")
    print(f"Menu: {menu_output}")
    
    # Save results to a text file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"restaurant_output_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("=== LangChain Restaurant Generator Output ===\n")
            file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Method: Manual Chain Management\n")
            file.write(f"Cuisine Type: {cuisine}\n")
            file.write("-" * 50 + "\n\n")
            file.write(f"Restaurant Name: {restaurant_name_output}\n\n")
            file.write(f"Menu Items: {menu_output}\n")
            file.write("\n" + "=" * 50)
        
        print(f"\n✅ Results saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving to file: {e}")


def initializeApp():
    llm = create_llm()
    restaurant_name_prompt_template = create_restaurant_name_prompt_template()
    restaurant_menu_prompt_template = create_restaurant_menu_prompt_template()
    return (llm, restaurant_name_prompt_template, restaurant_menu_prompt_template)

if __name__ == '__main__':
    """
    Main execution block that demonstrates two different LangChain approaches:
    
    1. Manual Chain Management (Choice 1):
       - Uses pipe operator to create chains
       - Manually handles data flow between chains
       - Gives full control over each step
       - Better for learning and debugging
    
    2. Sequential Chain (Choice 2):
       - Uses LangChain's SequentialChain class
       - Automatically manages data flow
       - More scalable for complex workflows
       - Built-in logging and error handling
    """
    print("Choose implementation:")
    print("1. Individual Chains (current implementation)")
    print("2. Sequential Chain")
    
    choice = input("Enter your choice (1 or 2): ")
    
    # Initialize all required components
    # These are shared between both implementations
    
    configuration = initializeApp()
    llm = configuration[0]
    restaurant_name_prompt_template = configuration[1]
    restaurant_menu_prompt_template = configuration[2]
    # Get cuisine input from user
    cuisine = input("Enter the cuisine type (e.g., Italian, Mexican, Chinese): ")

    # Route to appropriate implementation based on user choice
    if choice == "1":
        # Manual chaining approach - educational and gives full control
        main(cuisine=cuisine,
             llm=llm,
             restaurant_name_prompt=restaurant_name_prompt_template,
             restaurant_menu_prompt=restaurant_menu_prompt_template)
    elif choice == "2":
        # Sequential chain approach - more scalable and production-ready
        main_with_sequential_chain(cuisine=cuisine,
                                   llm=llm,
                                   restaurant_name_prompt=restaurant_name_prompt_template,restaurant_menu_prompt=restaurant_menu_prompt_template)
    else:
        print("Invalid choice. Running default implementation.")
        # Fallback to manual approach if invalid input
        main()