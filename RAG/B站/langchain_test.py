from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter





from langchain.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("..\data\Virtual_characters.pdf")
PDF_data=loader.load()


# 切分文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)



# 利用embedding模型建立向量数据库
persist_directory="db"
model_name = "all-MiniLM-L6-v2"
model_kwargs={"device":"cuda"}
embedding =HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs)
vectordb=Chroma.from_documents(documents=all_splits,embedding=embedding,persist_directory=persist_directory)


retriever = vectordb.as_retriever()
from transformers import AutoModelForCausalLM
model =AutoModelForCausalLM.from_pretrained('D:\Program Projects\Python Projects\DB-GPT\models\Qwen2-0.5B')
model=model.cuda()
qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)