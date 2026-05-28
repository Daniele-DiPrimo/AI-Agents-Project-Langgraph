from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from src.state import BlogState
from src.tools import blog_tools, blog_tools_wout_done

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
    """Analizza il prompt originale e popola lo stato."""
    messages = state.get("messages", [])
    if not messages:
        user_prompt = ""
    else:
        user_prompt = messages[-1].content
    
    system_prompt = (
        "Sei un router esperto per un blog universitario. Classifica la richiesta in "
        "NEWS per trattare argomenti di frontiera, TEORIA per trattare argomenti generici, ESERCIZI quando necessario. "
        "Popola inoltre lo stato con intent, macro_domain e specific_topic."
    )
    
    # Usiamo with_structured_output per forzare l'LLM a restituire un JSON perfetto
    structured_llm = classifier_llm.with_structured_output(IntentClassification)
    
    # Invocazione isolata per garantire l'amnesia ed evitare inquinamento della memoria
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {"intent": result.intent, "macro_domain": result.macro_domain, "specific_topic": result.specific_topic}

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