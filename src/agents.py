from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.structures import ClassificationSchema
from src.state import BlogState

import json
from langgraph.types import interrupt, Command
from langgraph.graph import END

# --- 1. SETUP MODELLI GROQ ---
classifier_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

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
        return {"intent": "Sconosciuto",
        "macro_domain": "Errore", 
        "specific_topic": "Nessun Input Fornito",
        "prompt_to_reasoner": "L'utente non ha fornito alcun input valido. Chiedi all'utente di specificare un argomento."
        }

    # 2. PROMPT CHIRURGICO (Blindato contro gli apostrofi)
    system_prompt = f"""Sei un classificatore chirurgico per un blog universitario.
    Devi analizzare ESATTAMENTE questa richiesta dell'utente: "{user_prompt_safe}"

    REGOLE DI COMPILAZIONE DEL JSON:
    - intent: Se la richiesta contiene "teoria", "spiega" o "come funziona", scrivi "Teoria". Se contiene "news", "notizie" o "novità", scrivi "News". Se contiene "esercizio", scrivi "Esercizio".
    - macro_domain: La materia generale (es. "C++", "Sistemi Operativi").
    - specific_topic: L'argomento preciso. Estrailo direttamente dal prompt.
    - prompt_to_reasoner: Trasforma la richiesta dell'utente in una direttiva chiara per il nodo successivo che scriverà l'articolo.."
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
            "specific_topic": result.specific_topic,
            "prompt_to_reasoner": result.prompt_to_reasoner
        }
    except Exception as e:
        print(f"⚠️ Errore API Groq nel Classifier: {e}")
        # Se Groq crasha (es. rate limit o failed generation), non facciamo crollare l'intero Studio
        fallback_prompt = f"Scrivi un articolo informativo riguardo il seguente argomento: {user_prompt_safe[:50]}"
        return {
            "intent": "Teoria", # Default sicuro
            "macro_domain": "Generico", 
            "specific_topic": user_prompt_safe[:50], # Usa l'input dell'utente come topic temporaneo
            "prompt_to_reasoner": fallback_prompt
        }


# =====================================================
# INIZIO NODO DI SCRITTURA
# =====================================================
# NODO WRITER UNIFICATO
def writer_node(state: BlogState) -> dict:
    """
    Nodo scrittore unificato: produce l'articolo finale o gli esercizi in Markdown.
    Estrae i dati direttamente dallo stato ed esegue il routing in base all'intent.
    """
    print("✍️ [Writer] Preparazione stesura...")

    # 1. Estrazione diretta dei dati dallo stato
    intent             = state.get("intent", "Unknown")
    macro_domain       = state.get("macro_domain", "Unknown")
    specific_topic     = state.get("specific_topic", "Unknown")
    prompt_to_reasoner = state.get("prompt_to_reasoner", "Scrivi in base al materiale fornito.")
    research_material  = state.get("research_material", "")

    # Fallback di sicurezza essenziale
    if not research_material:
        raise ValueError(
            "[Writer] research_material è vuoto. "
            "Il completeness_evaluator non ha completato correttamente il suo lavoro o c'è un errore nel passaggio di stato."
        )

    # 2. Routing interno del System Prompt basato sull'intent
    if intent.lower() == "esercizio":
        print("🎓 -> Modalità: Professore (Esercizi)")
        sys_prompt = f"""Sei un Professore Universitario esperto in '{macro_domain}'.
        Il tuo compito è creare esercizi pratici e stimolanti sull'argomento '{specific_topic}'.

        REGOLE FONDAMENTALI E VINCOLI:
        1. Basati ESCLUSIVAMENTE sul materiale di ricerca che ti verrà fornito dall'utente.
        2. NON inventare formule, sintassi o concetti non presenti nel testo fornito.
        3. Crea 2 o 3 esercizi di difficoltà crescente (indica il livello: Base / Intermedio / Avanzato).
        4. Per ogni esercizio scrivi chiaramente la TRACCIA, usando dati numerici e scenari realistici presenti nel materiale.
        5. Fornisci la SOLUZIONE DETTAGLIATA per ogni esercizio con spiegazione dei passaggi.
        6. Usa Markdown per separare nettamente tracce e soluzioni.
        7. Scrivi rigorosamente in ITALIANO."""
            
    else:
        print(f"📝 -> Modalità: Redattore (Tipo: {intent})")
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
        """

    research_material_msg = HumanMessage(content=f"Ecco il materiale di ricerca validato su cui devi basare l'articolo:\n{research_material}")

    # 3. Assemblaggio e invocazione
    llm_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Istruzioni specifiche per la redazione:\n{prompt_to_reasoner}"),
        research_material_msg
    ]

    final_draft = writer_llm.invoke(llm_messages)

    print("✅ [Writer] Stesura completata.")
    return {"final_article": final_draft.content}
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