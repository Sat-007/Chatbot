# Sat_CHATBOT
-----------------------------------------------------------------------------------------------------------------------

# PROJECT_NAME: SatChatBOT, smart chatbot to answer the questions in a domain 


# Description:  
 This is a chatbot which I have built here will perform a vector search on every question asked by the user, It will use the information from the vector database provided by the pdf files to answer the questions from the user, if any questions asked out of the domain, the answer will be I dont know

# Tech Stack Used: Python, Ollama models (nomic-embed-text and llama3)
# Operating System: Windows 11 (64 bit)
# Gradio is used for WEB UI interface as this is light weight

# Issues Faced
## Initially I wanted to use the OpenAI model, I Managed to write a working code for OpenAI model but due to billing restrictions I couldn't use OPENAI. 
## To over come this use, I installed Ollama and launched it locally on my machine
## To launch just the Ollama model, I used docker along with webUI to test whether the model was running

# Installation:
- Python Version: 3.7+
- pip install langchain_community langchain_text_splitters langchain_openai langchain_chroma gradio python-dotenv pypdf requests


# Setup:
- Install Ollama from https://ollama.com/
- After installation please run the following commands on cmd : 1) ollama pull nomic-embed-text 2) ollama pull llama3
- Fork my repo
- Run the database_generator.py. 
  This script will take the PDF files, cut it in small chunks and ingest it in the database (ChromaDB) 
- Run the sat_chatbot.py, we can see a link the terminal
  Open this link by CTRL + CLICK or by manually copy pasting the link in the browser of your choice

