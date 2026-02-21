import os
import pandas as pd
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Optional

class FeedbackState(TypedDict):
    user_id: str
    username: str
    responses: dict
    formatted_feedback: str
    context: Optional[str]
    persona: Optional[str]
    improvements: Optional[str]


#config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

rag_llm = ChatGroq(model="llama-3.3-70b-versatile")

base_path = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(base_path, "feedback_data", "user_feedback.csv")

print(f"Debug - Script location: {base_path}")
print(f"Debug - Full feedback file path: {FEEDBACK_FILE}")
print(f"Debug - Directory exists: {os.path.exists(os.path.dirname(FEEDBACK_FILE))}")

#create directory if missing
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

#save feedback
#df.to_csv(FEEDBACK_FILE, index=False)

QUESTIONS = [
    "What do you like most about this community?",
    "What are your critiques about this community?",
    "What improvements would you suggest?",
    "How active are you in events? (Rarely, Sometimes, Often)",
    "On a scale of 1-10, how would you rate this community?",
    "What are your future ambitions from this community?",
    "What, in your opinion, should the main goals of this club be?",
    "How together do you feel the community is towards a common ideology?",
    "How would you improve the togetherness in this club given your previous response?"
]

#set up logging level

logging.basicConfig(level=logging.INFO)

#STATIC data load (for training-style RAG)

def load_training_context():

    base_path = os.path.dirname(os.path.abspath(__file__))  
    training_dir = os.path.join(base_path, "training_data")

    training_files = [
        os.path.join(training_dir, "training_qna.txt"),
        os.path.join(training_dir, "example_personas.txt")
    ]

    docs = []

    for fname in training_files:
        if os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = [chunk.strip() for chunk in content.split('---') if chunk.strip()]
                for chunk in chunks:
                    docs.append(Document(page_content=chunk))
                logging.info(f"✅ Loaded {len(chunks)} chunks from {fname}")
        else:
            logging.warning(f"{fname} not found!")

    return docs


docs = load_training_context()
if not docs:
    raise ValueError("No valid training documents found. Check training_qna.txt and example_personas.txt.")

#test the embedding model
try:

    logging.info("Testing embeddings on first document chunk...")

    test_embed = embed_model.embed_documents([docs[0].page_content])

    if not test_embed or len(test_embed[0]) == 0:
        raise ValueError("Embedding model returned empty result.")

    logging.info("Embedding model test passed.")

except Exception as e:
    logging.error(f"Embedding model test failed: {e}")
    raise

#build vectorstore
try:
    vectorstore = Chroma.from_documents(docs, embedding=embed_model, collection_name="persona_feedback_training")
    logging.info("Chroma vectorstore successfully initialized.")

except Exception as e:
    logging.error(f"Failed to initialize Chroma vectorstore: {e}")
    raise


#ALL THE AGENT NODES BASED ON LANG GRAPH

#1. Feedback Collector (already done outside)

def feedback_collector(state):
    return state

#2. Retriever Agent

def make_context_node(retriever):
    def retrieve_context(state: FeedbackState):
        user_input = state["formatted_feedback"]
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        state["context"] = context
        return state
    return retrieve_context

#3. Persona Generator Agent

def persona_generator(state):
    SYSTEM_PROMPT = """
    You are a club AI assistant. Use the feedback and context to generate a realistic user persona.
    Always include:
    - Persona name
    - Estimated age (the range is 18-22)
    - Their goals, behaviour and interests related to the club.
    - Noted suggestions for improvement
    If not enough context, make a general but realistic persona. If you find something weird or out of context, point it out.
    ```
    {context}
    ```
    """
    HUMAN_PROMPT = "Generate a user persona from the following feedback:\n{input}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])
    chain = prompt | rag_llm | StrOutputParser()
    state["persona"] = chain.invoke({"context": state["context"], "input": state["formatted_feedback"]})
    return state

#4. Improvement Analyzer Agent

def improvement_analyzer(state):
    IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "Extract any constructive suggestions the user made for improving the club. Output in bullet points."),
        ("human", "{input}")
    ])
    chain = IMPROVE_PROMPT | rag_llm | StrOutputParser()
    state["improvements"] = chain.invoke({"input": state["formatted_feedback"]})
    return state

#createLangGraph 
graph = StateGraph(FeedbackState)
graph.add_node("collect_feedback", feedback_collector)
graph.add_node("retrieve_context", make_context_node(vectorstore.as_retriever()))
graph.add_node("generate_persona", persona_generator)
graph.add_node("analyze_improvements", improvement_analyzer)

#createsEdges
graph.set_entry_point("collect_feedback")
graph.add_edge("collect_feedback", "retrieve_context")
graph.add_edge("retrieve_context", "generate_persona")
graph.add_edge("generate_persona", "analyze_improvements")
graph.add_edge("analyze_improvements", END)

app = graph.compile()

#Execution Wrapper
def format_feedback(responses):
    return "\n".join([f"Q: {q}\nA: {a}" for q, a in responses.items()])

def run_feedback_pipeline(user_id, username, responses):
    formatted = format_feedback(responses)
    initial_state = {"user_id": user_id, "username": username, "responses": responses, "formatted_feedback": formatted}
    result = app.invoke(initial_state)

    data = {
        "User ID": user_id,
        "Username": username,
        **responses,
        "Persona": result.get("persona", ""),
        "Suggestions": result.get("improvements", "")
    }

    #debug stuff
    print(f"Debug - Current working directory: {os.getcwd()}")
    print(f"Debug - Feedback file path: {FEEDBACK_FILE}")
    print(f"Debug - Feedback file absolute path: {os.path.abspath(FEEDBACK_FILE)}")
    print(f"Debug - Directory exists: {os.path.exists(os.path.dirname(FEEDBACK_FILE))}")

    try:
        df = pd.read_csv(FEEDBACK_FILE)
        print("Existing CSV file loaded successfully")
    except FileNotFoundError:
        print("Creating new CSV file.")
        df = pd.DataFrame(columns=["User ID", "Username"] + QUESTIONS + ["Persona", "Suggestions"])

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    
    #error handling for the file save
    try:
        #to make sure the directory exists before saving
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        
        df.to_csv(FEEDBACK_FILE, index=False)
        print(f"Feedback saved successfully to: {FEEDBACK_FILE}")
        print(f"File size: {os.path.getsize(FEEDBACK_FILE)} bytes")
        
    except PermissionError as e:
        print(f"Permission denied - cannot write to file: {e}")
        print("Try running as administrator or check folder permissions")
        
    except FileNotFoundError as e:
        print(f"Directory path not found: {e}")
        print("Check if the folder structure exists")
        
    except Exception as e:
        print(f"Unexpected error saving feedback: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try saving to current directory as backup
        backup_file = "user_feedback_backup.csv"
        try:
            df.to_csv(backup_file, index=False)
            print(f"Backup saved to current directory: {backup_file}")
        except Exception as backup_error:
            print(f"Even backup failed: {backup_error}")

    print("\nGenerated Persona:\n", data["Persona"])
    print("\nExtracted Suggestions:\n", data["Suggestions"])

#CLI Wrapper
def collect_feedback():

    user_id = input("Enter your College ID: ")
    username = input("Enter your Name: ")
    responses = {}

    print("\nFeedback Session Starting...\n")

    for q in QUESTIONS:
        responses[q] = input(f"{q}\n> ")

    run_feedback_pipeline(user_id, username, responses)

if __name__ == "__main__":
    collect_feedback()
