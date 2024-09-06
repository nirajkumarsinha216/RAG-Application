from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import tempfile
from Gemini_Api import API_KEY

warnings.filterwarnings("ignore")

#Loading Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=API_KEY,
                             convert_system_message_to_human=True,system_message="You are a Delivery Manager in a Software Company. Provide clear and insightful answers.")

st.title("Document Reader")
data = st.file_uploader("Choose a file")

if data is not None:

    #Saving file temporary to get the temporary file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(data.read())
        temp_file_path = temp_file.name

    # Load and split the PDF content
    pdf_loader = PyPDFLoader(temp_file_path)
    pages = pdf_loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=API_KEY)

    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":1})

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_index,
    return_source_documents=True

)
    question = st.text_input("Enter Your Query")
    result = qa_chain({"query": question})
    st.write(result["result"])