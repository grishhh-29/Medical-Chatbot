from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import google.generativeai as genai
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def gemini_runnable(messages):
    # messages could be tuples like ('system', text) or objects with .content
    prompt_parts = []
    for m in messages:
        if isinstance(m, tuple):
            role, text = m
            prompt_parts.append(f"{role.upper()}: {text}")
        else:
            prompt_parts.append(m.content)
    prompt = "\n".join(prompt_parts)
    
    response = model.generate_content(prompt)
    return response.text

chatModel = RunnableLambda(gemini_runnable)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),        
    ]
)


question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])
    
    return render_template('chat.html')




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)