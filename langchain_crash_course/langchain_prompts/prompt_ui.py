from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

def main():
    print("Main Called")
    st.header('Research Tool')

    llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", 
                          temperature=0, 
                          api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
    
    
    paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

    style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

    length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


    template = load_prompt('template.json')

    if st.button('Submit'):
        chain = template | llm
        response = chain.invoke({
            'paper_input':paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
        st.write(response.content)



if __name__ == '__main__':
    main()