from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from src.state import BlogState
from src.tools import blog_tools, blog_tools_wout_done
import json
from langgraph.types import interrupt, Command

# --- 1. SETUP MODELLI GROQ ---
classifier_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
reasoning_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
react_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# --- 2. LOGICA CLASSIFICATORE ---
class IntentClassification(BaseModel):
    intent: str = Field(
        description="L'intento dell'utente. Scegli tra: 'News', 'Teoria', 'Esercizio'"
    )
    macro_domain: str = Field(
        description="Il dominio generale o la materia dell'argomento "
    )
    specific_topic: str = Field(
        description="Il focus specifico e dettagliato dell'articolo"
    )


# --- 3. NODO CLASSIFICATORE ---
def classifier_node(state: BlogState):
    """Analizza il prompt originale in modo infallibile e popola lo stato."""
    
    # 1. ESTRAZIONE CORAZZATA RIGOROSA
    user_prompt = "Nessun prompt riconosciuto." # Fallback di base
    
    # Se Studio passa i messaggi tramite la UI della chat
    if "messages" in state and state["messages"]:
        user_prompt = state["messages"][-1].content
    # Se Studio passa l'input tramite il campo original_prompt
    elif "original_prompt" in state and state.get("original_prompt"):
        user_prompt = state["original_prompt"]

    # Pulizia: rimuoviamo accenti o apostrofi strani che fanno impazzire Groq
    user_prompt_safe = str(user_prompt).replace("'", " ").replace('"', " ").strip()

    # Se l'input è vuoto, fermiamo il modello prima che provi a fare JSON a caso
    if not user_prompt_safe:
        return {"intent": "Sconosciuto", "macro_domain": "Errore", "specific_topic": "Nessun Input Fornito"}

    # 2. PROMPT CHIRURGICO (Blindato contro gli apostrofi)
    system_prompt = f"""Sei un classificatore chirurgico per un blog universitario.
Devi analizzare ESATTAMENTE questa richiesta dell'utente: "{user_prompt_safe}"

REGOLE DI COMPILAZIONE DEL JSON:
- intent: Se la richiesta contiene "teoria", "spiega" o "come funziona", scrivi "Teoria". Se contiene "news", "notizie" o "novità", scrivi "News". Se contiene "esercizio", scrivi "Esercizio".
- macro_domain: La materia generale (es. "C++", "Sistemi Operativi").
- specific_topic: L'argomento preciso. Estrailo direttamente dal prompt.
- IMPORTANTE: Rimuovi QUALSIASI apostrofo, virgoletta o carattere speciale (es. $, \, /) dai valori che generi nel JSON. Usa solo lettere e spazi.
"""
    
    try:
        # Usiamo with_structured_output per forzare il JSON
        structured_llm = classifier_llm.with_structured_output(IntentClassification)
        
        # Invochiamo il modello
        result = structured_llm.invoke([SystemMessage(content=system_prompt)])
        
        return {
            "intent": result.intent, 
            "macro_domain": result.macro_domain, 
            "specific_topic": result.specific_topic
        }
    except Exception as e:
        print(f"⚠️ Errore API Groq nel Classifier: {e}")
        # Se Groq crasha (es. rate limit o failed generation), non facciamo crollare l'intero Studio
        return {
            "intent": "Teoria", # Default sicuro
            "macro_domain": "Generico", 
            "specific_topic": user_prompt_safe[:50] # Usa l'input dell'utente come topic temporaneo
        }



# --- 4. NODO REASONER ---
def reasoner_node(state: BlogState):
    """L'agente principale: autonomo nella ricerca, ma con divieto assoluto di scrittura."""
    intent = state.get("intent", "Sconosciuto")
    macro_domain = state.get("macro_domain", "Sconosciuto")
    specific_topic = state.get("specific_topic", "Sconosciuto")
    
    # Prompt AGENTICO: gli diamo le regole di scelta e gli inibiamo la saggistica
    system_prompt = f"""Sei un Ricercatore Data-Miner per un blog universitario.
Il tuo UNICO compito è raccogliere dati grezzi sull'argomento: '{specific_topic}' (Dominio: {macro_domain}, Intento: {intent}).

REGOLE ASSOLUTE E INVIOLABILI:
1. TU NON SEI LO SCRITTORE. È SEVERAMENTE VIETATO scrivere l'articolo finale, usare formattazione Markdown o inventare tag XML come <done>...</done>.
2. Devi limitarti ESCLUSIVAMENTE a estrarre informazioni tramite i tool a tua disposizione.
3. Quando ritieni di aver raccolto abbastanza contesto nei messaggi precedenti, chiama il tool 'done' passando 'completato' e FERMATI. Non generare testo libero.

CRITERI DI SCELTA DEI TOOL (AGENTIC REASONING):
- Analizza l'argomento.
- Usa TAVILY SEARCH per: concetti teorici, definizioni, tutorial di programmazione (es. liste, dizionari) e notizie generiche.
- Usa ARXIV SEARCH SOLO ED ESCLUSIVAMENTE per: paper accademici, AI di frontiera, scoperte scientifiche complesse. NON usare ArXiv per la teoria informatica di base.
- Formula le query sempre in INGLESE.
"""
    
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message] + state["messages"]

    tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
    ha_gia_cercato = len(tool_messages) > 0

    # Diamo all'agente tutti i tool e lasciamo che applichi i "Criteri di scelta"
    if not ha_gia_cercato: 
        available_tools = blog_tools_wout_done
    else: 
        available_tools = blog_tools
    
    # parallel_tool_calls=False previene chiamate multiple non necessarie
    reasoning_agent = reasoning_llm.bind_tools(available_tools, tool_choice="any", parallel_tool_calls=False)
    response = reasoning_agent.invoke(messages)
    
    return {"messages": [response]}


# --- 5. NODO WRITER ---
def writer_node(state: BlogState):
    """L'Agente Scrittore: prende i dati raccolti e impagina il Markdown."""
    intent = state.get("intent", "Sconosciuto")
    macro_domain = state.get("macro_domain", "Sconosciuto")
    specific_topic = state.get("specific_topic", "Sconosciuto")
    
    system_prompt = f"""Sei l'autore principale di un blog tecnico universitario.
    Scrivi un articolo di tipo '{intent}' sulla materia '{macro_domain}' sull'argomento '{specific_topic}'.
    Basati ESCLUSIVAMENTE sulle informazioni e sulle ricerche presenti nella cronologia della conversazione.
    Scrivi in ITALIANO usando Markdown pulito. Inizia con un H1 e usa sezioni ben divise.
    NON inventare nulla che non sia nei dati forniti dal ricercatore.
    """
        
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message] + state["messages"]
    
    # QUI NESSUN TOOL! Modello libero di scrivere testo puro
    response = react_llm.invoke(messages)
    
    return {"messages": [response]}


# --- 6. NODO EXERCISES WRITER --- 
def exercises_writer_node(state: BlogState):
    """L'Agente Professore: prende i dati raccolti e formula esercizi pratici."""
    macro_domain = state.get("macro_domain", "Sconosciuto")
    specific_topic = state.get("specific_topic", "Sconosciuto")
    
    system_prompt = f"""Sei un Professore Universitario esperto in {macro_domain}.
    Il tuo compito è creare degli esercizi stimolanti sull'argomento '{specific_topic}'.
    Basati sui dati e sui concetti raccolti dal ricercatore nella cronologia.

    REGOLE DI FORMATTAZIONE:
    1. Crea 2 o 3 esercizi di difficoltà crescente.
    2. Per ogni esercizio, scrivi chiaramente la TRACCIA.
    3. Se l'esercizio richiede dati numerici o scenari, usa quelli trovati dal ricercatore per renderli realistici.
    4. Fornisci la SOLUZIONE DETTAGLIATA per ogni esercizio, spiegando i passaggi logici.
    5. Usa il Markdown per separare bene le tracce dalle soluzioni.
    """
    
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message] + state["messages"]
    
    # Usiamo lo stesso modello potente dello scrittore, ma con un'identità diversa
    response = react_llm.invoke(messages)
    
    return {"messages": [response]}

from langgraph.types import interrupt, Command
from langgraph.graph import END

def human_review_node(state: BlogState) -> Command:
    """Nodo corazzato che mette in pausa il grafo per far revisionare l'articolo all'utente.
    Resiste a input errati, stringhe non formattate e JSON rotti da LangGraph Studio."""
    
    # 1. Recuperiamo l'articolo appena scritto
    ultimo_messaggio = state["messages"][-1].content
    
    # 2. Creiamo il pacchetto da inviare all'esterno (al terminale o a LangGraph Studio)
    richiesta_revisione = {
        "azione": "revisione_articolo",
        "anteprima": ultimo_messaggio[:500] + "...\n[Testo troncato per l'anteprima]"
    }
    
    # 3. 🛑 IL GRAFO SI CONGELA QUI! 🛑
    raw_resume_value = interrupt([richiesta_revisione])
    
    # --- 🛡️ INIZIO BLOCCO DI SICUREZZA (LA CORAZZA) 🛡️ ---
    # A. Estraiamo il valore se è nascosto in una lista
    if isinstance(raw_resume_value, list) and len(raw_resume_value) > 0:
        risposta_umana = raw_resume_value[0]
    else:
        risposta_umana = raw_resume_value

    # B. Se la UI (o l'utente) ha mandato una stringa di testo invece di un JSON
    if isinstance(risposta_umana, str):
        try:
            # Proviamo a convertirla in un dizionario formattato
            risposta_umana = json.loads(risposta_umana)
        except json.JSONDecodeError:
            # Se la conversione fallisce (es. l'utente ha scritto testo libero nel box)
            # Analizziamo la stringa manualmente per non far crashare il nodo
            testo_libero = risposta_umana.lower()
            if "approva" in testo_libero:
                risposta_umana = {"tipo": "approva"}
            elif "modifica" in testo_libero:
                # Salviamo l'intero testo come feedback
                risposta_umana = {"tipo": "modifica", "feedback": risposta_umana}
            else:
                risposta_umana = {"tipo": "annulla"}
    
    # C. Se alla fine di tutto NON è ancora un dizionario, apriamo il paracadute
    if not isinstance(risposta_umana, dict):
        risposta_umana = {"tipo": "annulla"}
    # --- 🛡️ FINE BLOCCO DI SICUREZZA 🛡️ ---

    # 4. IL GRAFO SI SCONGELA E PRENDE UNA DECISIONE SICURA
    # Usiamo .get() così se la chiave non esiste ci restituisce "annulla" invece di un errore
    azione = risposta_umana.get("tipo", "annulla")

    if azione == "approva":
        # Se approva, andiamo diretti alla fine
        return Command(goto=END)
        
    elif azione == "modifica":
        # Recuperiamo il feedback (se per caso l'utente non l'ha inserito, mettiamo un testo di default)
        feedback_testo = risposta_umana.get("feedback", "L'utente ha richiesto una revisione generale senza specificare dettagli.")
        
        feedback_msg = {
            "role": "user", 
            "content": f"L'utente ha rifiutato la bozza e ha richiesto questa modifica: {feedback_testo}. Riscrivi l'articolo tenendone conto."
        }
        
        # Scegliamo dove rimandarlo in base all'intent originale
        intent = state.get("intent", "").lower()
        ritorno = "exercises_writer" if intent == "esercizio" else "writer"
        
        return Command(
            goto=ritorno, 
            update={"messages": [feedback_msg]} # Aggiorniamo lo stato con il feedback
        )
        
    else:
        # Fallback (es. l'utente ha inviato 'annulla' o dati incomprensibili)
        return Command(goto=END)