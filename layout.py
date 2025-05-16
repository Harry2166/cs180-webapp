import streamlit as st

lorem_ipsum = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """

prediction_classes = {
    0: "risk",
    1: "neutral",
    2: "opportunity"
}

def section(title: str, text: str = lorem_ipsum):
    """
    This function creates a "section" of text that is basically a title with a bunch of text underneath it

    Parameters:
    -----------
    title : str 
            The title of the section; Make sure to add the appropriate number of "#"
    text  : str
            The text to be put under the section; By default, it is set to lorem ipsum. It is recommended to pass "" if you want to have no text underneath the title. It is also recommended to store all the text in a multiline string variable and just pass that instead.  
    """
    st.write(f"{title}")
    st.write(f"{text}")

