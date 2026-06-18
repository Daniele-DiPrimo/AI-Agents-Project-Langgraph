import os
from dotenv import load_dotenv
import requests
from google import genai
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from src.structures import SearchSchema
from src.mcp_client import call_mcp_tool

# Carica il file .env per la chiave API
load_dotenv()

embedder = genai.Client()

# Tool di Ricerca Web
tavily = TavilySearch(
    max_results=1,               
    search_depth="advanced",     # esegue ricerche multiple
    include_raw_content=False,
)

# Tool Python Code Executor
# Permette all'LLM di scrivere ed eseguire codice per risolvere esercizi tecnici
python_tool = PythonREPLTool()

@tool(args_schema=SearchSchema)
def search_tool(giustificazione:str, query: str) -> str:
    """
    Cerca informazioni sul Web. 
    Usa questo tool per TEORIA GENERALE, notizie, tutorial.
    - giustificazione: DEVI spiegare brevemente PERCHÉ stai facendo questa specifica ricerca.
    - query: Le parole chiave in inglese.
    """
    risposta = tavily.invoke({"query": query})
    return str(risposta)

@tool(args_schema=SearchSchema)
def search_semantic_scholar(giustificazione: str, query: str) -> str:
    """
    Cerca paper accademici e scientifici tramite Semantic Scholar. 
    USA QUESTO TOOL per la ricerca di frontiera.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 1,
        "fields": "title,abstract,authors,year,url"
    }

    headers = {}
    api_key= os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if api_key: 
        headers["x-api-key"] = api_key
    else:
        headers["User-Agent"] = "UniAgent2.0"

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data.get("total", 0) == 0 or "data" not in data:
            return "Nessun paper rilevante trovato per questa query."
            
        risultati = []
        for paper in data["data"]:
            autori = ", ".join([a["name"] for a in paper.get("authors", [])])
            risultati.append(
                f"Titolo: {paper.get('title')}\n"
                f"Anno: {paper.get('year')}\n"
                f"Autori: {autori}\n"
                f"Abstract: {paper.get('abstract')}\n"
                f"URL: {paper.get('url')}\n"
                "---"
            )
        return "\n".join(risultati)
        
    except Exception as e:
        return f"Errore durante la ricerca accademica: {str(e)}"
    
@tool
async def ricerca_krag_unificata(query_ricerca: str) -> str:
    """
    Usa questo strumento per cercare informazioni nel Knowledge Graph.
    Passa una domanda chiara o un concetto per cui scrivere l'articolo (es. "Scrivi un articolo sugli integrali").
    Il sistema restituirà automaticamente i testi, i concetti teorici collegati, 
    le affermazioni chiave da supportare e i nomi dei file PDF da citare.
    """
    try:
        print(f"🔍 [Planner] Esecuzione Ricerca K-RAG per: '{query_ricerca}'")
        
        # Calcola l'embedding della query
        risultato_embedding = embedder.models.embed_content(
            model="gemini-embedding-2",
            contents=query_ricerca
        )
        vettore = risultato_embedding.embeddings[0].values
        
        # Chiama il tool K-RAG tramite MCP
        risultato_testo = await call_mcp_tool(
            tool_name="ricerca_ibrida_krag",
            arguments={
                "vettore_query": vettore,
                "top_k": 5
            }
        )
        
        return risultato_testo
                
    except Exception as e:
        print(f"❌ [Errore Tool K-RAG]: {str(e)}")
        return f"Errore durante la ricerca nel Knowledge Graph: {str(e)}"


blog_tools = [search_semantic_scholar, search_tool, ricerca_krag_unificata]