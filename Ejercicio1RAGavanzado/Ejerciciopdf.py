from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, SystemMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore


from langgraph.checkpoint.memory import InMemorySaver

#hola
CHROMA_DIR = "./Ejercicio1RAGavanzado/chroma_db"
COLLECTION_NAME = "recetas_pdf"

def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar. Que sea el mismo con el que vectorizamos los documentos!
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings

def crear_retriever(vectorstore: Chroma):



    padre = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap= 200) 
    hijo = RecursiveCharacterTextSplitter(chunk_size= 400, chunk_overlap= 50)


    #Almacen padre
    DOCUMENTOS_PADRE = LocalFileStore("./documentos_padre")

    #Almacen hijo

    vectorstore = Chroma(
        embedding_function= embeddings,
        persist_directory= CHROMA_DIR,
        collection_name= COLLECTION_NAME
    )

    retriever = ParentDocumentRetriever(
        child_splitter= hijo,
        parent_splitter= padre,
        vectorstore= vectorstore,
        docstore= create_kv_docstore(DOCUMENTOS_PADRE)
    )
    
    return retriever

def conectar_crhroma():
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )

    print("..Chromadb listp...")

    retriever = crear_retriever(vectorstore)

    print("...Chromadb integrado...")

    return retriever



@tool()
def obtener_info_rag(pregunta: str):

    """ Esta herramienta se encarga de conectar con ChromaDB, hacer la consulta y devolver la información relevante para el agente. 
    Todas las respuestas deben de responderse utilizando esta herramienta exclusivamente y sin inventar informacion. 
    Si la información no se encuentra en la base de datos, se debe responder con un mensaje claro indicando que no se encontró información relevante. """

    retriever = conectar_crhroma()
    return retriever.invoke(pregunta)


modelo = ChatOllama(model="gemma4:26b")
agente = create_agent(
    
    model=modelo,
    tools=[obtener_info_rag],
    checkpointer=InMemorySaver(),

    system_prompt = """
    Eres un agente basado en RAG (Retrieval Augmented Generation) diseñado para responder preguntas utilizando información
    relevante obtenida de una base de datos.
    Eres un cocinero experto en recetas canarias, y deves de responder de manera cercana y amigable, como si fueras un amigo que comparte sus conocimientos culinarios.
    Debes de responder a las preguntas de manera clara y sencilla, utilizando la información relevante que obtienes de la base de datos. Si no encuentras información relevante, debes responder con un mensaje claro indicando que no se encontró información relevante.""")





def hablarConChat(agente):
    while (prompt := input("> ")) != "end":
        for paso in agente.stream(
            {
                "messages": [HumanMessage(prompt)]
            },
            stream_mode="values",
            config={"configurable": {"thread_id": "Gabrielito"}}
        ):
            ultimo_mensaje = paso["messages"][-1]

            hayRazonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"):
                hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hayRazonamiento:
                print("\n=== PENSANDO ===")
                print(hayRazonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()


hablarConChat(agente)