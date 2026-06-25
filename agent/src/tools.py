import os
from dotenv import load_dotenv
import requests
from google import genai
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from src.structures import SearchSchema
from src.mcp_client import call_mcp_tool

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
def search_tool(giustificazione:str, query: str) -> dict:
    """
    Cerca informazioni sul Web. 
    Usa questo tool per TEORIA GENERALE, notizie, tutorial.
    - giustificazione: DEVI spiegare brevemente PERCHÉ stai facendo questa specifica ricerca.
    - query: Le parole chiave in inglese.
    """
    risposta = tavily.invoke({"query": query})
    return risposta

@tool(args_schema=SearchSchema)
def search_semantic_scholar(giustificazione: str, query: str) -> dict: # <-- Cambiato da str a dict
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
        
        # Se non ci sono risultati, ritorna il dizionario vuoto standard
        if data.get("total", 0) == 0 or "data" not in data:
            return {"query": query, "results": []}
            
        risultati = []
        for paper in data["data"]:

            print(f"PAPER SEMANTICH {paper}")
            autori = ", ".join([a["name"] for a in paper.get("authors", [])])
            
            # Assembliamo il testo utile da passare all'LLM valutatore
            content = (
                f"Titolo: {paper.get('title')}\n"
                f"Anno: {paper.get('year')}\n"
                f"Autori: {autori}\n"
                f"Abstract: {paper.get('abstract')}"
            )
            
            # Creiamo il formato atteso: { "url": ..., "content": ... }
            risultati.append({
                "source": paper.get("url", "URL Mancante"),
                "content": content
            })
            
        # Ritorna il formato esatto per il source_evaluator
        return {
            "query": query,
            "results": risultati
        }
        
    except Exception as e:
        print(f"❌ [Errore Tool Semantic Scholar]: {str(e)}")
        return {
            "query": query, 
            "results": []
        }
    

@tool
async def analisi_gap_contenuti(materia_specifica: str = "") -> dict:
    """
    Usa questo strumento quando devi pianificare nuovi articoli o capire cosa manca nel blog.
    Interroga la memoria storica del Knowledge Graph per estrarre tutti gli articoli già scritti,
    i concetti trattati e il loro livello di dettaglio (claims).
    
    - materia_specifica: (Opzionale) Stringa esatta della materia su cui filtrare il report (es. "Fisica", "Matematica"). 
                         Lasciare vuoto "" per analizzare l'intero grafo.
    """
    try:
        print(f"📊 [Planner] Chiamata a MCP per analisi dei content gap. Filtro materia: '{materia_specifica}'")
        
        # Gestiamo il fallback della stringa vuota o None per i parametri MCP
        arguments = {}
        if materia_specifica and materia_specifica.strip():
            arguments["materia_specifica"] = materia_specifica.strip()

        # Chiama il tool registrato sul server MCP tramite FastMCP
        risultato_testo = await call_mcp_tool(
            tool_name="ricerca_topic_gap",
            arguments=arguments
        )
        
        # Ritorniamo una struttura coerente con gli altri tool di ricerca (es. ricerca_krag_unificata)
        # in modo che il source_evaluator o i nodi successivi del grafo ricevano lo stesso formato.
        return {
            "query": materia_specifica if materia_specifica else "Tutte le materie",
            "results": [
                {
                    "source": "Knowledge_Graph_Coverage_Report",
                    "content": risultato_testo
                }
            ]
        }
                
    except Exception as e:
        print(f"❌ [Errore Tool Analisi Gap]: {str(e)}")
        return {
            "query": materia_specifica, 
            "results": []
        }
    

@tool
async def tool_ricerca_neo4j(embedded_title: list[float], top_k: int = 3) -> list[str]:
    """
    Interroga il Knowledge Graph in Neo4j tramite MCP usando un vettore di similarità.
    Restituisce concetti correlati, affermazioni chiave e metadati degli articoli.
    """
    try:
        print(f"🔍 [Neo4j] Chiamata a MCP per ricerca vettoriale.")
        risultato = await call_mcp_tool(
            tool_name="neo4j_search",
            arguments={
                "embedded_title": embedded_title,
                "top_k": top_k
            }
        )
        return risultato
    except Exception as e:
        print(f"❌ [Errore Tool Neo4j]: {str(e)}")
        return []

@tool
async def tool_ricerca_rag(queries: list[str], subject: str, top_k: int = 3) -> dict:
    """
    Interroga il Vector Database (ChromaDB) tramite MCP per estrarre il testo dei documenti 
    basandosi su una serie di query testuali.
    """
    try:
        print(f"📚 [RAG] Chiamata a MCP per ricerca documenti su materia: {subject}")
        risultato = await call_mcp_tool(
            tool_name="rag_search",
            arguments={
                "queries": queries,
                "subject": subject,
                "top_k": top_k
            }
        )
        return risultato
    except Exception as e:
        print(f"❌ [Errore Tool RAG]: {str(e)}")
        return {"query": " | ".join(queries), "results": []}


blog_tools = [search_semantic_scholar, search_tool]