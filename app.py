import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Set the page configuration
st.set_page_config(page_title="Content Summarizer", page_icon="📝")

# Apply custom styling with background image and gradient
st.markdown("""
    <style>
    body {
        background: url('https://via.placeholder.com/1920x1080.png?text=Background+Image') no-repeat center center fixed;
        background-size: cover;
        color: #3a3a3a;
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    .header img {
        width: 100px;
    }
    .header h1, .header h3 {
        color: #3a3a3a; /* Change text color here */
    }
    .stButton>button {
        background-color: #f3a1a1;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #e09e9e;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .stTextInput>div>input:focus {
        border-color: #f3a1a1;
    }
    </style>
    <div class="header">
        <img src="https://via.placeholder.com/100x50.png?text=Logo" alt="Header Image">
        <h1>📝 Summarize Content from YouTube & Web</h1>
        <h3>Your go-to tool for quick and concise summaries</h3>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Setup Your Summarizer")
    groq_api_key = st.text_input("Groq API Key", type="password")
    generic_url = st.text_input("Enter URL")

# Prompt template for summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to trigger summarization
if st.button("📝 Summarize Now"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or Website).")
    else:
        try:
            with st.spinner("Summarizing content..."):
                # Load content from the URL
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0"})
                docs = loader.load()

                # Summarization chain
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
