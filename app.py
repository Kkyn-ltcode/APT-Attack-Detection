from nav import apt
import streamlit as st

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def run():
    state = st.session_state

    apt.write(state)

if __name__ == '__main__':
    run()
