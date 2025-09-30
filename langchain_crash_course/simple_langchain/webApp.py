import streamlit as st
from main import main_with_sequential_chain, initializeApp

st.title("Restaurant Name and Menu generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ("Indian", "Chinese", "Mexican", "Arabic", "Italian"))

if cuisine:
    configuration = initializeApp()
    llm = configuration[0]
    restaurant_name_prompt_template = configuration[1]
    restaurant_menu_prompt_template = configuration[2]

    result = main_with_sequential_chain(cuisine=cuisine,
                               llm=llm,
                               restaurant_name_prompt=restaurant_name_prompt_template,
                               restaurant_menu_prompt= restaurant_menu_prompt_template)
    st.header(result['restaurant_name'])
    menu_items = result['menu_items'].split(",")
    for item in menu_items:
        st.write(f"- {item}")
