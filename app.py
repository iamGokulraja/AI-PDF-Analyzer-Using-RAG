import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
import os

def chunkText(text):
    chunks = []
    chunk_size = 500
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def createEmbeddings(chunks):
    return [model.encode(chunk).tolist() for chunk in chunks]

def ConnectToDB():
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port = os.getenv("DB_PORT")
    )
    register_vector(conn)
    return conn

def replaceDocument(chunks , vectors):
    con = ConnectToDB()
    cur = con.cursor()
    try:
        cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")

        for chunk, vector in zip(chunks, vectors):
            cur.execute(
            """
            INSERT INTO documents (chunk_text, embedding) VALUES (%s, %s)
            """, (chunk, vector))
        con.commit()
    finally:
        cur.close()
        con.close()

def findRelatedVector(questionVector):
    con = ConnectToDB()
    cur = con.cursor()
    try:
        cur.execute(
            """ 
            SELECT chunk_text FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT 3 
            """, (questionVector,))
        return cur.fetchall()
        
    finally:
        cur.close()
        con.close()

def initModel():
    load_dotenv()
    apikey = os.getenv("API_KEY")
    client = OpenAI(api_key=apikey , base_url="https://openrouter.ai/api/v1")
    return client

def generatePrompt(relatedChunks, question):
    prompt = f"""
    "You are a helpful assistant. Answer the question only based on the context provided .
    
    Context :
    {relatedChunks}

    Question : {question}
    """
    return prompt


model = load_model()
conn = ConnectToDB()
client = initModel()

if "processed" not in st.session_state:
    st.session_state.processed = False

st.title("PDF Question Answering App")
uploadedFile = st.file_uploader("Upload a PDF file", type="pdf")
if st.button('Upload'):
    #if file upload 
    if uploadedFile is not None:
        try:
            pdfReader = PdfReader(uploadedFile)
            text = ""
            for page in pdfReader.pages:
                text += page.extract_text()
            #st.write(text)  # Display the extracted text for debugging
            chunks = chunkText(text)
            vectors = createEmbeddings(chunks)
            #st.write(vectors)  # Display the embeddings for debugging

            try:
                replaceDocument(chunks, vectors)
                st.success("PDF processed and data stored in the database.")
                st.session_state.processed = True
            except Exception as e:
                st.error(f"An error occurred while storing data in the database: {e}")
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
    else:
        st.warning("Please upload a PDF file.")

if st.session_state.processed:
    st.success("You can now ask questions based on the content.")
    question = st.text_input("Ask a question about the PDF content:")
    if st.button('Ask'):
        if question.strip() :
            questionVector = createEmbeddings([question])[0]
            # st.write(questionVector)
            result = findRelatedVector(questionVector)
            prompt = generatePrompt(result, question)
            try:
                response = (client.chat.completions.create(model = "openrouter/free" ,
                    messages = [
                        {
                          'role' : 'user' ,
                          'content' : prompt
                        }
                    ]))
                st.write("Response:")
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
        else:
            st.warning("Please enter a question.")
                            
