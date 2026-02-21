import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#load api key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#load the model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_llm = ChatGroq(model="llama-3.3-70b-versatile")

questions = [
    "What do you like most about this community?",
    "What improvements would you suggest?",
    "How active are you in events? (Rarely, Sometimes, Often)",
    "On a scale of 1-10, how would you rate this community?",
    "What are your future ambitions from this community?",
    "How was your day?"]
    
FEEDBACK_FILE = "feedback.csv"

def generate_persona(responses):
    formatted_responses = "\n".join([f"Q: {q}\nA: {a}" for q, a in responses.items()])
    
    RAG_SYSTEM_PROMPT = """
    You are an AI assistant for Enigma, the Computer Science Club, that summarizes user responses into a short persona.
    Generate a persona name and an estimated age always.
    Use the retrieved context to generate a user persona based on the feedback.
    Base the user persona in such a way that their goals, behaviour and interests are related to the community.
    Have a separate deduction on what the user sees for improvement.
    ```
    {context}
    ```
    If you don't have enough context, generate a general persona.
    """

    RAG_HUMAN_PROMPT = f""" Generate a concise user persona based on the following responses: {formatted_responses}"""

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT)
    ])
    
    #set up the vector database
    document = Document(page_content=formatted_responses)
    vectorstore = Chroma.from_documents([document], embedding=embed_model, collection_name="feedback_personas")
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    
    rag_pipeline = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | RAG_PROMPT
        | rag_llm
        | StrOutputParser()
    )
    
    return rag_pipeline.invoke("Summarize this user's persona")

def feedbackInput(user_id, username, responses):
    data = {"User ID": user_id, "Username": username}
    data.update({q: responses.get(q, "") for q in questions})
    
    persona = generate_persona(responses)
    data["Persona"] = persona
    
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["User ID", "Username"] + questions + ["Persona"])
    
    new_entry = pd.DataFrame([data])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

def collect_feedback():
    user_id = input("Enter your User ID: ")
    username = input("Enter your Username: ")
    responses = {}
    
    print("\nFeedback Session Starting...\n")
    for question in questions:
        response = input(f"{question}\n> ")
        responses[question] = response
    
    feedbackInput(user_id, username, responses)
    
    print("\nThanks for your feedback! Here's what you said:\n")
    for q, a in responses.items():
        print(f"{q}\n{a}\n")
    
    print("\nGenerated Persona:\n")
    print(generate_persona(responses))

def print_feedback():

    try:
        df = pd.read_csv(FEEDBACK_FILE)
        if df.empty:
            print("No feedback has been collected yet.")
        else:
            print("\n--- Feedback Data ---\n")
            print(df)

    except FileNotFoundError:
        print("Feedback file not found.")

def print_responses():
    try:
        df = pd.read_csv(FEEDBACK_FILE)

        if df.empty:
            print("No feedback has been collected yet.")
            return
        
        print("=== Feedback Responses by User ===\n")

        for i in range(len(df)):
            print("User:", df.iloc[i, 0])  
            print("-" * 40)
            
            for question in df.columns[:]:  
                print(f"Q) {question}")
                print(f"A) {df.iloc[i][question]}")
                print()
        
        print("=" * 40)


    except FileNotFoundError:
        print("Feedback file not found.")


if __name__ == "__main__":
    collect_feedback()
    #print_responses()
    #print_feedback()       