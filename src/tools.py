import os
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
import requests

# Carica il file .env per la chiave API
load_dotenv()

# 2. Inizializza il tool usando TavilySearch
blog_tools = [
    TavilySearch(max_results=1)
]
# Tool 1: Ricerca Web (obbligatorio da specifiche)
# max_results=3 evita di saturare il contesto dell'LLM con troppi dati
_tavily_nascosto = TavilySearch(max_results=1)

# Tool 2: Python Code Executor (uno dei tool "custom")
# Permette all'LLM di scrivere ed eseguire codice per risolvere esercizi tecnici
python_tool = PythonREPLTool()


# --- SCHEMI PYDANTIC (Il pugno di ferro) ---
class SearchSchema(BaseModel):
    giustificazione: str = Field(
        description="OBBLIGATORIO: Spiega nel dettaglio il tuo ragionamento logico e PERCHÉ stai facendo questa ricerca."
    )
    query: str = Field(
        description="La query di ricerca in inglese."
    )

class DoneSchema(BaseModel):
    giustificazione: str = Field(
        description="OBBLIGATORIO: Spiega perché ritieni di avere tutti i dati necessari per passare alla stesura dell'articolo."
    )
    ready: str = Field(
        description="Scrivi 'completato'."
    )


#aspetto la api_key
@tool(args_schema=SearchSchema)
def search_tool(giustificazione:str, query: str) -> str:
    """
    Cerca informazioni sul Web. 
    Usa questo tool per TEORIA GENERALE, notizie, tutorial o se ArXiv fallisce.
    - giustificazione: DEVI spiegare brevemente PERCHÉ stai facendo questa specifica ricerca.
    - query: Le parole chiave in inglese.
    """
    
    # Richiami il tool nascosto passandogli la query. 
    # L'LLM non ha modo fisico di alterare i parametri qui dentro.
    risposta = _tavily_nascosto.invoke({"query": query})
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
    
    try:
        response = requests.get(url, params=params)
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

@tool(args_schema=DoneSchema)
def done(giustificazione:str, ready: str) -> str:
    """
    Chiama questo tool quando hai finito la ricerca.
    - giustificazione: Spiega perché ritieni di avere tutti i dati necessari per l'articolo.
    - ready: Scrivi "completato".
    """
    return "Ricerca completata. Passo alla stesura."

blog_tools_wout_done = [search_semantic_scholar, search_tool]
blog_tools = [search_tool, python_tool, search_semantic_scholar, done]