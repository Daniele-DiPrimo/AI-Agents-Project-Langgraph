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
async def ricerca_krag_unificata(query: str) -> dict:
    """
    Usa questo strumento per cercare informazioni nel Knowledge Graph.
    Passa una domanda chiara o un concetto per cui scrivere l'articolo (es. "Scrivi un articolo sugli integrali").
    Il sistema restituirà automaticamente i testi, i concetti teorici collegati, 
    le affermazioni chiave da supportare e i nomi dei file PDF da citare.
    """
    try:
        print(f"🔍 [Planner] Esecuzione Ricerca K-RAG per: '{query}'")
        
        # Calcola l'embedding della query
        """risultato_embedding = embedder.models.embed_content(
            model="gemini-embedding-2",
            contents=query
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
        
        return risultato_testo"""
        
        return {
            "query": query,
            "results":
                [
                    {
                        "source": "Teorema_di_Shannon.pdf",
                        "content": "Il Teorema del Campionamento di Nyquist-Shannon rappresenta il fondamento teorico di tutta la tecnologia digitale moderna.\nFormulato inizialmente da Harry Nyquist e successivamente sviluppato e dimostrato da Claude Shannon nel 1949,\nquesto teorema definisce la condizione minima affinché un segnale analogico possa essere convertito in digitale\nsenza alcuna perdita di informazione e successivamente ricostruito in modo perfetto.\n\nL'enunciato fondamentale stabilisce che la frequenza di campionamento (fs) deve essere strettamente superiore\nal doppio della frequenza massima (fmax) contenuta nello spettro del segnale originale.\nLa formula matematica che esprime questo vincolo è semplicemente: fs > 2 * fmax.\nLa metà della frequenza di campionamento (fs / 2) viene comunemente chiamata 'Frequenza di Nyquist'.\n\nSe la frequenza di campionamento è inferiore a questa soglia critica, si verifica un fenomeno distorsivo\nnoto come 'aliasing'. Durante l'aliasing, le frequenze superiori alla metà della frequenza di campionamento\nsi sovrappongono alle frequenze più basse, creando falsi segnali (o 'alias') che rendono impossibile\nla ricostruzione accurata dell'onda analogica di partenza.\n\nPer evitare questo problema nei sistemi reali, prima del processo di conversione analogico-digitale (ADC),\nil segnale viene fatto passare attraverso un filtro passa-basso analogico, chiamato 'filtro anti-aliasing'.\nQuesto filtro ha il compito di tagliare drasticamente tutte le frequenze superiori alla frequenza di Nyquist.\n\nOltre al campionamento, Claude Shannon ha introdotto la celebre formula per la capacità di canale,\nnota come legge di Shannon-Hartley: C = B * log2(1 + SNR).\nIn questa formula, C rappresenta la capacità del canale in bit al secondo, B è la banda passante in Hertz,\ne SNR è il rapporto segnale-rumore (Signal-to-Noise Ratio).\n\nQuesta equazione definisce il limite teorico massimo di dati che possono essere trasmessi in modo affidabile\nattraverso un canale di comunicazione affetto da rumore termico gaussiano bianco.\n\nLe applicazioni di questi concetti sono ovunque: dalla codifica audio (i CD usano 44.1 kHz per coprire i 20 kHz udibili),\nalle comunicazioni mobili (4G, 5G), fino allo streaming video e alla compressione dei dati.\nSenza le intuizioni di Shannon, l'era dell'informazione digitale non sarebbe mai potuta esistere."
                    },
                    {
                        "source": "Equazioni_Matematiche.pdf",
                        "content": "In matematica, un'equazione costituisce uno degli strumenti più potenti e versatili mai concepiti.\nEssa è definita formalmente come un'uguaglianza tra due espressioni algebriche che contiene\nuna o più quantità variabili, denominate incognite, solitamente indicate con lettere come x, y, z.\n\nRisolvere un'equazione significa determinare l'insieme dei valori che, se sostituiti alle incognite,\nrendono l'uguaglianza logica e matematica vera. Tali valori prendono il nome di soluzioni o radici.\n\nLa struttura di un'equazione è divisa in due parti principali separate dal simbolo di uguale (=):\nil 'primo membro', situato a sinistra, e il 'secondo membro', situato a destra.\nLe equazioni possono essere classificate in base a criteri differenti, primo fra tutti il grado.\nIl grado di un'equazione polinomiale intera corrisponde al massimo esponente con cui compare l'incognita.\n\nLe equazioni di primo grado (o lineari) ammettono, nel campo dei numeri reali, una sola soluzione.\nLe equazioni di secondo grado (o quadratiche) possono avere due soluzioni reali, una sola, o nessuna reale\n(spostandosi nel campo dei numeri complessi). Il Teorema Fondamentale dell'Algebra\ngarantisce che un'equazione polinomiale di grado n ha esattamente n soluzioni nel campo complesso.\n\nUn'ulteriore classificazione distingue le equazioni in base alla loro natura intrinseca:\n- Equazioni algebriche: contengono solo operazioni algebriche elementari sulle incognite.\n- Equazioni trascendenti: coinvolgono funzioni trigonometriche, logaritmiche o esponenziali.\n- Equazioni differenziali: mettono in relazione una funzione incognita con le sue derivate.\n- Equazioni integrali: dove la funzione incognita compare sotto il segno di integrale.\n\nI metodi di risoluzione variano drasticamente a seconda del tipo di equazione affrontata.\nSi va dalle formule analitiche chiuse (come la formula risolutiva delle equazioni di secondo grado)\nfino a sofisticati algoritmi di calcolo numerico utilizzati dai computer per equazioni non lineari complesse.\n\nNel contesto delle scienze applicate, le equazioni fungono da modelli matematici per descrivere la realtà.\nDalle leggi di Maxwell per l'elettromagnetismo all'equazione di Schrödinger per la meccanica quantistica,\nl'intera fisica teorica è formulata e compresa attraverso il linguaggio rigoroso delle equazioni."
                    },
                    {
                        "source": "La_Forza_in_Fisica.pdf",
                        "content": "Nel dominio della fisica classica, il concetto di forza è di fondamentale importanza.\nLa forza viene definita operativamente come una grandezza fisica vettoriale che misura\nl'intensità dell'interazione meccanica tra due corpi o tra un corpo e il campo circostante.\n\nEssendo una grandezza vettoriale, una forza non è caratterizzata solo da un valore numerico (modulo),\nma possiede intrinsecamente anche una direzione, un verso e un punto di applicazione specifico.\nL'effetto di una forza può essere dinamico (modifica il moto) o statico (provoca deformazioni).\n\nLo studio sistematico delle forze ha inizio con Sir Isaac Newton e i suoi tre principi della dinamica:\n1. Il principio di inerzia: un corpo mantiene il suo stato di quiete o di moto rettilineo uniforme\n   se la somma delle forze esterne agenti su di esso è rigorosamente pari a zero.\n2. Il principio fondamentale: la forza risultante applicata a un corpo è direttamente proporzionale\n   all'accelerazione impressa, secondo la celeberrima formula vettoriale: F = m * a.\n3. Il principio di azione e reazione: a ogni forza esercitata da un corpo su un altro\n   corrisponde una forza uguale in modulo e direzione, ma opposta in verso, esercitata dal secondo sul primo.\n\nNel Sistema Internazionale di unità di misura, l'unità della forza è il Newton (N),\ndefinito come la forza necessaria per imprimere a una massa di 1 kg l'accelerazione di 1 m/s².\nLo strumento utilizzato per misurare sperimentalmente l'intensità delle forze è il dinamometro.\n\nLa fisica moderna riduce tutte le interazioni osservabili in natura a quattro forze fondamentali:\n- La forza di gravità: attrattiva, agisce su tutte le masse ed ha raggio d'azione infinito.\n- La forza elettromagnetica: agisce sulle cariche elettriche, responsabile della struttura atomica.\n- La forza nucleare forte: tiene uniti i quark all'interno dei protoni e dei neutroni.\n- La forza nucleare debole: responsabile di alcuni processi di decadimento radioattivo.\n\nL'evoluzione della fisica ha modificato la percezione della forza: per Albert Einstein,\nla gravità non è una forza tradizionale, ma la conseguenza geometrica della curvatura dello spaziotempo.\nNella teoria quantistica dei campi, invece, le forze vengono spiegate attraverso lo scambio\ndi particelle mediatrici della forza stessa, chiamate bosoni di gauge (come i fotoni o i gluoni)."
                    }
                ]
        }
                
    except Exception as e:
        print(f"❌ [Errore Tool K-RAG]: {str(e)}")
        return {}
    
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


blog_tools = [search_semantic_scholar, search_tool]