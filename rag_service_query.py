import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
import numpy as np
import joblib

# Load the trained model
oos_model = joblib.load('oos_model.pkl')

CHROMA_DB_PATH = "database"

PROMPT_TEMPLATE = """
Answer the question based on the following context don't exceed the context :

{context}

---

Answer the following question within the above context : {question}
"""

PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING= """

I will provide you with a context enclosed in triple backticks (). Your task is to analyze the information within this context and answer questions exclusively based on the provided details.

Guidelines:

1.Strict Adherence: Base your responses solely on the information presented within the context. Do not introduce external knowledge or assumptions.
2.Conciseness and Relevance: Craft answers that are brief, factual, and directly pertinent to the questions asked.
3.Information Gaps: If a question cannot be answered due to insufficient information in the context, respond with: "The context does not provide this information."


{context}
```

With Respect to the Instructions given, Answer the following question : {question}

"""

RAW_PROMPT_TEMPLETE = " {context} Provide a Short Answer for the following question with your knowledge : {question} "

def extract_features(query_text, db, k=5):
    results = db.similarity_search_with_score(query_text, k=k)
    scores = [score for _, score in results]

    # Extract features
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    score_variance = np.var(scores)
    num_docs = len(scores)

    return [avg_score, max_score, score_variance, num_docs], results

def query_rag(query_text: str):
    # Configure the Database
    embedding_function = get_embedding_function()
    # Initialize Chroma database
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    # Extract features and results
    features, results = extract_features(query_text, db)
    features_array = np.array(features).reshape(1, -1)

    # Predict OOS
    is_oos = oos_model.predict(features_array)[0]

    if is_oos:
        response_text = "I'm sorry, but your query seems to be out of scope for my knowledge base."
        return {
            'question': query_text,
            'response': response_text,
            'scores': features,
            'sources': []
        }

    # Extract the context text from search results
    context_text = "---".join([doc.page_content for doc, _score in results])

    # Generate a prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Populate the prompt template with context and question
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the Ollama model
    model = get_bedrock_model()

    # Get the response from the Ollama model based on prompt
    response_text = model.invoke(prompt)

    # Extract sources from search results
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Format the response with sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    # Return response_text
    return {
        'question': query_text,
        'response': response_text,
        'sources': sources,
        'scores': features
    }

def query_rag_with_prompt_engineering(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    features, results = extract_features(query_text, db)
    features_array = np.array(features).reshape(1, -1)

    # Predict OOS
    is_oos = oos_model.predict(features_array)[0]

    if is_oos:
        response_text = "I'm sorry, but your query seems to be out of scope for my knowledge base."
        return {
            'question': query_text,
            'response': response_text,
            'scores': features,
            'sources': []
        }

    context_text = "---".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return {
        'question': query_text,
        'response': response_text,
        'sources': sources,
        'scores': features
    }

def raw_query(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_templete = ChatPromptTemplate.from_template(RAW_PROMPT_TEMPLETE)
    prompt = prompt_templete.format(question=query_text,context=context_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"
    return {
        'question': query_text,
        'response': response_text,
        'sources': "N/A",
        'scores': "N/A"
    }