from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.structures import ClassificationSchema, FullSourcesEvaluationSchema, CompletenessEvaluationSchema
from src.state import BlogState, ReasonerState
from src.tools import blog_tools
import json
from langgraph.types import interrupt, Command
from langgraph.graph import END

# --- 1. SETUP MODELLI GROQ ---
classifier_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
planner_completeness_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
react_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
source_evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

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
        structured_llm = classifier_llm.with_structured_output(ClassificationSchema)
        
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


# =====================================================
# INIZIO NOODI DEL SOTTOGRAFO DI REASONING [PLANNER, SOURCE_EVALUATOR, COMPLETENESS_EVALUATOR]
# =====================================================

# PLANNER
def planner_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile del piano esecutivo per la ricerca.
    Decide quali tool chiamare e genera le query.
    Scrive solo in: messages (loop ReAct interno).
    """
 
    intent        = state.get("intent", "NotSpecified")
    macro_domain  = state.get("macro_domain", "NotSpecified")
    specific_topic = state.get("specific_topic", "NotSpecified")
 
    planner_llm_w_tools = planner_completeness_llm.bind_tools(
        blog_tools,
        tool_choice="auto",
        parallel_tool_calls=False
    )
 
    sys_prompt = f"""Sei l'Agente Ricercatore Capo. Il tuo scopo è chiamare i tool corretti per cercare informazioni su:
    - Intent: {intent}
    - Dominio: {macro_domain}
    - Topic specifico: {specific_topic}
    
    REGOLE DI SELEZIONE DEI TOOL:
    - Usa tavily per ricerche generiche (News, Teoria di base)
    - Usa semantic_scholar per ricerche accademiche avanzate (Teoria approfondita, paper)
    
    REGOLE OPERATIVE:
    - Ti è vietato scrivere informazioni sull'argomento: devi SOLO cercare e inoltrare i risultati.
    - Se pensi di avere già tutto il necessario, rispondi senza usare tool."""
 
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    answer = planner_llm_w_tools.invoke(messages)
 
    # Scrive SOLO nei messages — è l'unico nodo che lo fa legittimamente
    return {"messages": [answer]}
    

# SOURCE EVALUATOR
def source_evaluator_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile della valutazione granulare delle fonti trovate.
    Legge: messages (ultimi risultati tool)
    Scrive: approved_sources, sources_evaluated
    NON tocca messages.
    """
 
    print("🔍 [Source Evaluator] Valutazione granulare delle fonti...")
 
    structured_source_evaluator_llm = source_evaluator_llm.with_structured_output(
        FullSourcesEvaluationSchema,
        method="json_mode"
    )
 
    sys_prompt = """Sei un ispettore delle fonti.
    Analizza i risultati delle ultime ricerche (i messaggi di tipo 'tool').
    Per ogni informazione trovata, assegna un punteggio da 0.0 a 1.0.
    Sii spietato con le informazioni generiche, fuori tema o allucinate.
    Usa ESATTAMENTE questa struttura JSON:
    {
    "judgments": [
        {
        "id_source": "url o identificatore",
        "rate": 0.0,
        "reasoning": "motivazione"
        }
    ],
    "need_new_search": false
    }"""
    
    # Legge solo gli ultimi messaggi (risultati dei tool più recenti)
    messages = [SystemMessage(content=sys_prompt)] + state["messages"][-5:]
    judgment: FullSourcesEvaluationSchema = structured_source_evaluator_llm.invoke(messages)
 
    ACCEPTANCE_THRESHOLD = 0.5
 
    approved_sources = [
        {"id_source": v.id_source, "rate": v.rate, "reasoning": v.reasoning}
        for v in judgment.judgments
        if v.rate >= ACCEPTANCE_THRESHOLD
    ]
    not_approved_sources = [
        v for v in judgment.judgments
        if v.rate < ACCEPTANCE_THRESHOLD
    ]
 
    print(f"   -> Fonti analizzate: {len(judgment.judgments)} | "
          f"Approvate: {len(approved_sources)} | "
          f"Bocciate: {len(not_approved_sources)}")
 
    if judgment.need_new_search or len(approved_sources) == 0:
        # Nessuna fonte valida — il coverage_evaluator lo rileverà da sources_evaluated=False
        reasons = ", ".join([v.reasoning for v in not_approved_sources])
        print(f"   -> ⚠️ Fonti insufficienti. Motivi: {reasons}")
        return {
            "approved_sources": [],
            "sources_evaluated": False
            # Nessun messaggio: sarà il coverage_evaluator a istruire il planner
        }
    else:
        # Fonti valide trovate — le salviamo in un campo strutturato
        existing = state.get("approved_sources", [])
        return {
            "approved_sources": existing + approved_sources,
            "sources_evaluated": True
        }
    

# COMPLETENESS_EVALUATOR
def completeness_evaluator_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile della valutazione della completezza complessiva del materiale.
    Legge: approved_sources, sources_evaluated, intent, macro_domain, specific_topic
    Scrive: is_complete, research_material (se completo), messages (solo se serve istruire il planner)
    """
 
    completeness_structured_llm = planner_completeness_llm.with_structured_output(
        CompletenessEvaluationSchema
    )
 
    intent         = state.get("intent", "NotSpecified")
    macro_domain   = state.get("macro_domain", "NotSpecified")
    specific_topic = state.get("specific_topic", "NotSpecified")
    approved_sources = state.get("approved_sources", [])
    sources_evaluated = state.get("sources_evaluated", False)
    iterations     = state.get("iterations", 0)
 
    # Caso 1: source_evaluator ha già segnalato fonti insufficienti
    # Istruiamo il planner a ritentare con parole chiave diverse
    if not sources_evaluated:
        warning = HumanMessage(
            content="⚠️ ATTENZIONE: Tutte le fonti recenti sono state scartate. "
                    "Usa parole chiave o tool diversi per la prossima ricerca."
        )
        return {
            "is_complete": False,
            "iterations": iterations + 1,
            "messages": [warning]
        }
 
    # Caso 2: valutiamo la copertura complessiva sul materiale approvato
    sys_prompt = f"""Sei il Chief Editor di un blog tecnico universitario.
    Valuta se il materiale raccolto è sufficiente per scrivere un contenuto di qualità.
    
    OBIETTIVO: articolo di tipo [{intent}], materia [{macro_domain}], argomento [{specific_topic}].
    
    CRITERI DI SUFFICIENZA:
    - Definizioni principali presenti
    - Concetti chiave coperti
    - Almeno un esempio pratico (se applicabile)
    NON serve un'enciclopedia: se il cuore dell'argomento è coperto, approva.
    
    REGOLE OPERATIVE:
    - NON fare domande all'utente.
    - Se manca il nucleo fondamentale, scrivi in 'missing_info' 2-3 parole chiave in inglese per la prossima ricerca."""
 
    # Passiamo il materiale approvato come contesto, NON tutta la history dei messages
    sources_summary = "\n".join([
        f"- Fonte {s['id_source']} (score {s['rate']}): {s['reasoning']}"
        for s in approved_sources
    ])
    human_msg = HumanMessage(content=f"Materiale raccolto finora:\n{sources_summary}")
 
    answer: CompletenessEvaluationSchema = completeness_structured_llm.invoke([
        SystemMessage(content=sys_prompt),
        human_msg
    ])
 
    if answer.is_complete:
        # Costruisce research_material strutturato da passare al BlogState → writer
        research_material = f"# Materiale di ricerca validato\n\n"
        research_material += f"**Tipo articolo:** {intent} | **Dominio:** {macro_domain} | **Topic:** {specific_topic}\n\n"
        for s in approved_sources:
            research_material += f"## Fonte: {s['id_source']} (affidabilità: {s['rate']:.1f}/1.0)\n"
            research_material += f"{s['reasoning']}\n\n"
 
        print(f"✅ [Coverage Evaluator] Materiale sufficiente. Fonti approvate: {len(approved_sources)}")
        return {
            "is_complete": True,
            "research_material": research_material,
            "iterations": iterations + 1
        }
    else:
        # Istruisce il planner con le info mancanti tramite messages
        instruction = HumanMessage(
            content=f"INFO MANCANTI: Il revisore ha detto che manca: '{answer.missing_info}'. "
                    f"Fai un'altra ricerca mirata su questo."
        )
        print(f"🔄 [Coverage Evaluator] Materiale incompleto. Manca: {answer.missing_info}")
        return {
            "is_complete": False,
            "iterations": iterations + 1,
            "messages": [instruction]
        }

# =====================================================
# FINE NOODI DEL SOTTOGRAFO DI REASONING [PLANNER, SOURCE_EVALUATOR, COMPLETENESS_EVALUATOR]
# =====================================================


# =====================================================
# INIZIO NODI DI SCRITTURA (HELPER_WRITER - WRITER - EXERCISES WRITER)
# =====================================================

# HELPER CONDIVISO
def _build_writer_context(state: BlogState) -> tuple[str, str, str, str]:
    """
    Estrae dal BlogState i dati necessari ai writer.
    Restituisce: (intent, macro_domain, specific_topic, research_material)
    
    Con il nuovo stato, research_material è già un campo strutturato pulito —
    nessun bisogno di filtrare i messages.
    """
    intent          = state.get("intent", "Unknown")
    macro_domain    = state.get("macro_domain", "Unknown")
    specific_topic  = state.get("specific_topic", "Unknown")
    research_material = state.get("research_material", "")
 
    if not research_material:
        # Fallback di sicurezza — non dovrebbe mai accadere con il nuovo stato
        raise ValueError(
            "[Writer] research_material è vuoto. "
            "Il coverage_evaluator non ha completato correttamente il suo lavoro."
        )
 
    return intent, macro_domain, specific_topic, research_material
 
 
# NODO WRITER
def writer_node(state: BlogState) -> dict:
    """
    Nodo scrittore: produce l'articolo finale in Markdown.
    Legge: research_material, intent, macro_domain, specific_topic, original_prompt
    Scrive: final_article
    NON tocca messages — il contesto è già pulito in research_material.
    """
    print("✍️ [Writer] Preparazione articolo...")
 
    intent, macro_domain, specific_topic, research_material = _build_writer_context(state)
 
    sys_prompt = f"""Sei l'autore principale di un blog tecnico universitario.
Scrivi un articolo di tipo '{intent}' sulla materia '{macro_domain}' sull'argomento '{specific_topic}'.
 
REGOLA FONDAMENTALE:
Basati ESCLUSIVAMENTE sul materiale di ricerca validato fornito qui sotto.
NON inventare informazioni, concetti o codice non presenti in questa sintesi.
 
REGOLE DI FORMATTAZIONE:
- Scrivi in ITALIANO con Markdown pulito
- Inizia con un titolo H1 chiaro e descrittivo
- Usa sezioni ben divise con H2 e H3
- Includi una sezione finale "## Fonti" con tutti gli URL del materiale
 
--- INIZIO MATERIALE DI RICERCA VALIDATO ---
{research_material}
--- FINE MATERIALE DI RICERCA VALIDATO ---"""
 
    llm_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=state.get(
            "original_prompt",
            "Scrivi l'articolo in base al materiale fornito."
        ))
    ]
 
    final_draft = react_llm.invoke(llm_messages)
 
    print("✅ [Writer] Articolo completato.")
    return {"final_article": final_draft.content}
 

# NODO EXERCISES WRITER
def exercises_writer_node(state: BlogState) -> dict:
    """
    Nodo professore: produce esercizi pratici strutturati in Markdown.
    Legge: research_material, macro_domain, specific_topic, original_prompt
    Scrive: final_article
    NON tocca messages — il contesto è già pulito in research_material.
    """
    print("🎓 [Exercises Writer] Preparazione esercizi...")
 
    intent, macro_domain, specific_topic, research_material = _build_writer_context(state)
 
    sys_prompt = f"""Sei un Professore Universitario esperto in '{macro_domain}'.
    Il tuo compito è creare esercizi pratici e stimolanti sull'argomento '{specific_topic}'.
    
    REGOLA FONDAMENTALE:
    Basati ESCLUSIVAMENTE sul materiale di ricerca validato fornito qui sotto.
    NON inventare formule, sintassi o concetti non presenti in questa sintesi.
    
    REGOLE DI FORMATTAZIONE:
    1. Crea 2 o 3 esercizi di difficoltà crescente (indica il livello: Base / Intermedio / Avanzato)
    2. Per ogni esercizio scrivi chiaramente la TRACCIA
    3. Usa dati numerici e scenari realistici presi dal materiale
    4. Fornisci la SOLUZIONE DETTAGLIATA per ogni esercizio con spiegazione dei passaggi
    5. Usa Markdown per separare nettamente tracce e soluzioni
    6. Scrivi in ITALIANO
 
    --- INIZIO MATERIALE DI RICERCA VALIDATO ---
    {research_material}
    --- FINE MATERIALE DI RICERCA VALIDATO ---"""
 
    llm_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=state.get(
            "original_prompt",
            "Genera gli esercizi in base al materiale fornito."
        ))
    ]
 
    final_exercises = react_llm.invoke(llm_messages)
 
    print("✅ [Exercises Writer] Esercizi completati.")
    return {"final_article": final_exercises.content}

# =====================================================
# FINE NODI DI SCRITTURA (HELPER_WRITER - WRITER - EXERCISES WRITER)
# =====================================================

# =====================================================
# INIZIO NODO HITL (HUMAN_REVIEW --> NODO / _PARSE_HUMAN_RESPONSE_ PER GESTIRE LA RISPOSTA)
# =====================================================

def human_review_node(state: BlogState) -> Command:
    """
    HITL: mette in pausa il grafo per la revisione umana dell'articolo.
    Legge: final_article, intent
    Azioni possibili: approva → END | modifica → writer/exercises_writer | annulla → END
    """

    final_article = state.get("final_article", "")

    if not final_article:
        # Fallback di sicurezza — non dovrebbe accadere con il nuovo stato
        print("⚠️ [Human Review] final_article vuoto, approvazione automatica.")
        return Command(goto=END)

    # 1. Pacchetto inviato all'interfaccia (LangGraph Studio o client)
    review_request = {
        "azione": "revisione_articolo",
        "anteprima": final_article[:500] + "...\n[Testo troncato per l'anteprima]",
        "istruzioni": "Rispondi con: {'tipo': 'approva'} | {'tipo': 'modifica', 'feedback': '...'} | {'tipo': 'annulla'}"
    }

    # 2. Il grafo si congela qui
    raw_resume_value = interrupt([review_request])

    # 3. Normalizzazione robusta della risposta
    human_response = _parse_human_response(raw_resume_value)

    # 4. Routing in base all'azione
    action = human_response.get("tipo", "annulla")

    if action == "approva":
        print("✅ [Human Review] Articolo approvato.")
        return Command(goto=END)

    elif action == "modifica":
        feedback = human_response.get(
            "feedback",
            "Revisione generale richiesta senza dettagli specifici."
        )
        print(f"🔄 [Human Review] Modifica richiesta: {feedback[:100]}...")

        intent = state.get("intent", "").lower()
        destination = "exercises_writer" if intent == "esercizio" else "writer"

        return Command(
            goto=destination,
            update={
                # Passa il feedback come HumanMessage — formato corretto per LangGraph
                "messages": [HumanMessage(
                    content=f"[FEEDBACK REVISORE UMANO] {feedback}. "
                            f"Riscrivi l'articolo tenendo conto di questo feedback. "
                            f"Il materiale di ricerca validato è già disponibile in research_material."
                )],
                # Pulisce l'articolo precedente — il writer lo sovrascriverà
                "final_article": ""
            }
        )

    else:
        print("🚫 [Human Review] Articolo annullato o risposta non riconosciuta.")
        return Command(goto=END)


def _parse_human_response(raw: any) -> dict:
    """
    Normalizza qualsiasi risposta umana in un dict con chiave 'tipo'.
    Gestisce: lista, stringa JSON, stringa libera, dict, tipo sconosciuto.
    """
    # Spacchetta lista
    if isinstance(raw, list) and len(raw) > 0:
        raw = raw[0]

    # È già un dict — caso ideale
    if isinstance(raw, dict):
        return raw

    # È una stringa — proviamo JSON, poi testo libero
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Analisi testo libero
        text = raw.lower().strip()
        if any(k in text for k in ["approva", "approv", "ok", "sì", "si", "yes"]):
            return {"tipo": "approva"}
        if any(k in text for k in ["modifica", "cambia", "rivedi", "no", "feedback"]):
            return {"tipo": "modifica", "feedback": raw}

    # Paracadute finale
    return {"tipo": "annulla"}