import os
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests
from src.structures import SearchSchema

# Carica il file .env per la chiave API
load_dotenv()

# Tool 1: Ricerca Web (obbligatorio da specifiche)
# max_results=3 evita di saturare il contesto dell'LLM con troppi dati
tavily = TavilySearch(max_results=1, search_depth="advanced")

# Tool 2: Python Code Executor (uno dei tool "custom")
# Permette all'LLM di scrivere ed eseguire codice per risolvere esercizi tecnici
python_tool = PythonREPLTool()

#aspetto la api_key
@tool(args_schema=SearchSchema)
def search_tool(giustificazione:str, query: str) -> str:
    """
    Cerca informazioni sul Web. 
    Usa questo tool per TEORIA GENERALE, notizie, tutorial.
    - giustificazione: DEVI spiegare brevemente PERCHÉ stai facendo questa specifica ricerca.
    - query: Le parole chiave in inglese.
    
    """
    
    # Richiami il tool nascosto passandogli la query. 
    # L'LLM non ha modo fisico di alterare i parametri qui dentro.
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
        "limit": 1, # Bastano 2 paper per non esaurire i token
        "fields": "title,abstract,authors,year,url" # Estraiamo solo la "polpa"
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


blog_tools = [search_semantic_scholar, search_tool]