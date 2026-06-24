from dotenv import load_dotenv
import json

from google import genai
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt, Command
from langgraph.graph import END
from src.tools import analisi_gap_contenuti
from src.mcp_client import call_mcp_tool

from src.structures import ClassificationSchema, EstrazioneMetadatiArticolo, PlannerSchema
from src.state import BlogState

from src.prompts import (
    get_classifier_prompt,
    get_planner_prompt,
    get_writer_exercise_prompt,
    get_writer_article_prompt,
    get_metadata_extractor_prompt
)

load_dotenv()

classifier_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
planner_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
embedder = genai.Client()



def classifier_node(state: BlogState):
    """Analizza il prompt originale e popola lo stato."""
    
    user_prompt = "Nessun prompt riconosciuto." # Fallback di base
    
    # Se Studio passa i messaggi tramite la UI della chat
    if "messages" in state and state["messages"]:
        user_prompt = state["messages"][-1].content
    # Se Studio passa l'input tramite il campo original_prompt
    elif "original_prompt" in state and state.get("original_prompt"):
        user_prompt = state["original_prompt"]

    # Pulizia: rimuoviamo accenti o apostrofi strani per evitare problemi di parsing o di interpretazione del prompt da parte di Groq
    user_prompt_safe = str(user_prompt).replace("'", " ").replace('"', " ").strip()

    # Se l'input è vuoto, fermiamo il modello prima che provi a fare JSON casuali
    if not user_prompt_safe:
        print("⚠️ [Classifier] Nessun prompt utente fornito. Terminazione automatica.")
        return Command(goto=END)

    system_prompt = get_classifier_prompt()
    
    try:
        # Usiamo with_structured_output per forzare il JSON
        structured_llm = classifier_llm.with_structured_output(ClassificationSchema)
        
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Istruzioni specifiche per la redazione inviate dal coordinatore:\n{user_prompt_safe}")])
        
        return {
            "intent": result.intent, 
            "subject": result.subject, 
            "specific_topic": result.specific_topic,
            "prompt_to_reasoner": result.prompt_to_reasoner
        }
    except Exception as e:
        print(f"⚠️ Errore API Groq nel Classifier: {e}")
        # Se Groq presenta errori (es. rate limit o failed generation)
        return Command(goto=END)

async def planner_node(state: BlogState):
    """
    Analizza lo stato e la memoria storica del grafo tramite il tool di gap analysis,
    quindi interroga l'LLM per generare una lista strutturata di suggerimenti editoriali.
    """
    print("🔮 [Planner] Avvio analisi dei content gap per pianificazione...")

    # 1. Recuperiamo l'eventuale materia (subject) corrente dallo stato per filtrare il grafo
    # Se non c'è, passiamo una stringa vuota per analizzare tutto l'ecosistema
    materia_corrente = state.get("subject", "")
    
    # 2. Chiamata asincrona al tool MCP esposto in LangChain
    try:
        report_tool = await analisi_gap_contenuti.ainvoke({"materia_specifica": materia_corrente})
        print(f"REPORT TOOL: {report_tool}")
        # Estraiamo il contenuto testuale del report generato dal server MCP
        contesto_grafo = report_tool["results"][0]["content"]
    except Exception as e:
        print(f"⚠️ [Planner] Errore durante il recupero del report dal grafo: {e}")
        contesto_grafo = "Nessun dato sul grafo disponibile a causa di un errore tecnico."

    # 3. Configurazione dell'LLM strutturato
    planner_llm_structured = planner_llm.with_structured_output(PlannerSchema)
    
    system_prompt = get_planner_prompt()
    
    # Costruiamo il messaggio per l'interazione umana includendo l'input dell'utente e lo stato del grafo
    prompt_utente = ""
    if "messages" in state and state["messages"]:
        prompt_utente = state["messages"][-1].content
    elif "original_prompt" in state:
        prompt_utente = state["original_prompt"]

    user_content = f""" Richiesta originale dell'utente: {prompt_utente}
                        STATO ATTUALE DEL KNOWLEDGE GRAPH (Articoli già scritti e concetti): {contesto_grafo}
                        Genera la lista di suggerimenti in formato strutturato basandoti sui dati reali del grafo sopra riportati."""

    try:
        # 4. Invocazione del modello
        risultato = await planner_llm_structured.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        
        # 5. Restituzione dello stato aggiornato. 
        # Di solito in LangGraph si aggiorna una lista di suggerimenti o si passa il controllo al nodo successivo.
        # Ritorniamo il dizionario che aggiornerà le chiavi dello stato.
        return {
            "suggestions": risultato.suggestions
        }

    except Exception as e:
        print(f"❌ [Planner] Errore durante la generazione dei suggerimenti con l'LLM: {e}")
        # Fallback sicuro in caso di errore dell'API
        return {
            "suggestions": []
        }
    
def hitl_planner_node(state: BlogState):
    """
    Prende il suggerimento corrente e lo isola nello stato.
    Il grafo si interromperà SUBITO DOPO questo nodo.
    """
    idx = state.get("current_suggestion_index", 0)
    suggestions = state.get("suggestions", [])

    if idx >= len(suggestions):
        print("🏁 [HITL] Tutti i suggerimenti sono stati esaminati.")
        return Command(goto=END)
    
    current_suggestion = suggestions[idx]

    review_request = {
        "azione": "accettazione stesura articolo",
        "current_suggestion": current_suggestion,
        "istruzioni": "Rispondi con: {'tipo': 'approva'} | {'tipo': 'annulla'}"
    }

    # Il grafo si congela qui
    raw_resume_value = interrupt([review_request])
    human_response = parse_human_response(raw_resume_value)
    action = human_response.get("tipo", "annulla")

    if action == "approva":
        print("✅ [Human Review] Articolo approvato. Passo il controllo alla stesura...")
        return Command(
            goto="reasoner_subgraph",
            update={
                "intent": current_suggestion.intent, 
                "subject": current_suggestion.subject, 
                "specific_topic": current_suggestion.specific_topic,
                "prompt_to_reasoner": current_suggestion.prompt_to_reasoner
            }
        )

    else:
        print(f"🔄 [Human Review] Articolo rifiutato.")

        return Command(
            goto="hitl_planner",
            update={
                "current_suggestion_index": idx + 1,
            }
        )

async def writer_node(state: BlogState) -> dict:
    """
    Genera l'articolo finale o gli esercizi in Markdown, includendo obbligatoriamente le citazioni e gli URL visitati.
    Estrae i dati dallo stato (unendo K-RAG e Web RAG) ed esegue il routing in base all'intent.
    """
    print("✍️ [Writer] Preparazione stesura con unificazione K-RAG e Web...")

    # 1. Estrazione dei dati dallo stato
    intent             = state.get("intent", "Unknown")
    subject            = state.get("subject", "Unknown")
    specific_topic     = state.get("specific_topic", "Unknown")
    prompt_to_reasoner = state.get("prompt_to_reasoner", "Scrivi in base al materiale fornito.")
    research_material  = state.get("research_material", "") # Materiale dal Web validato

    # Fallback di sicurezza essenziale: blocca se non c'è assolutamente nessuna informazione
    if not research_material:
        return {"final_article": "[Writer] Il sottografo di ricerca non ha estratto alcuna informazione utile, non è stato possibile scrivere l'articolo."}

    # 3. Routing interno del System Prompt basato sull'intent (Aggiornato per supportare entrambe le fonti)
    if intent.lower() == "esercizio":
        print("🎓 -> Modalità: Professore (Esercizi)")
        sys_prompt = get_writer_exercise_prompt(subject, specific_topic)
    else:
        print(f"📝 -> Modalità: Redattore (Tipo: {intent})")
        sys_prompt = get_writer_article_prompt(intent, subject, specific_topic)

    # Impacchettiamo il mega-testo unificato nel messaggio per l'LLM
    context_msg = HumanMessage(content=f"Materiale di riferimento totale per la redazione:\n{research_material}")

    # 4. Assemblaggio dei messaggi e invocazione
    llm_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Istruzioni specifiche per la redazione inviate dal coordinatore:\n{prompt_to_reasoner}"),
        context_msg
    ]

    # Se ci sono messaggi precedenti (feedback umano o richieste di rigenerazione), li aggiungiamo per contestualizzare la stesura
    if "messages" in state and state["messages"]:
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, 'content'):
            if "[TIPO: RIGENERA]" in last_msg.content:
                llm_messages.append(last_msg)
                
            elif "[TIPO: MODIFICA]" in last_msg.content:
                llm_messages.append(last_msg)

    final_draft = await writer_llm.ainvoke(llm_messages)
    testo_generato = final_draft.content
    print("✅ [Writer] Stesura completata con successo basata su K-RAG totale.")

    return {"final_article": testo_generato}



async def human_review_node(state: BlogState) -> Command:
    """
    HITL: mette in pausa il grafo per la revisione umana dell'articolo.
    Si occupa SOLO di raccogliere il feedback e decidere il routing.
    """
    final_article = state.get("final_article", "")

    if not final_article:
        print("⚠️ [Human Review] final_article vuoto, terminazione automatica.")
        return Command(goto=END)

    review_request = {
        "azione": "revisione_articolo",
        "anteprima": final_article,
        "istruzioni": "Rispondi con: {'tipo': 'approva'} | {'tipo': 'modifica', 'feedback': '...'} | {'tipo': 'annulla'} | {'tipo': 'rigenera'} | {'tipo': 'nuova ricerca'}"
    }

    # Il grafo si congela qui
    raw_resume_value = interrupt([review_request])
    human_response = parse_human_response(raw_resume_value)
    action = human_response.get("tipo", "annulla")

    if action == "approva":
        print("✅ [Human Review] Articolo approvato. Passo il controllo al nodo di salvataggio...")
        return Command(
            goto="save_article",
            update={
                "research_material": ""
            }
        )

    elif action == "modifica":
        feedback = human_response.get("feedback", "Revisione generale richiesta senza dettagli specifici.")
        print(f"🔄 [Human Review] Modifica richiesta: {feedback}...")

        return Command(
            goto="writer",
            update={
                "messages": [HumanMessage(
                    content=f"[TIPO: MODIFICA]\nFeedback del revisore umano {feedback}. "
                            f"Riscrivi l'articolo tenendo conto di questo feedback. "
                            f"Il materiale di ricerca validato è già disponibile in research_material."
                )],
                "final_article": ""
            }
        )

    elif action == "rigenera_testo":
        print("🔄 [Human Review] Rigenerazione del testo richiesta (stesso materiale).")
        return Command(
            goto="writer",
            update={
                "messages": [HumanMessage(
                    content="[TIPO: RIGENERA]\nIgnora la stesura precedente e scrivi una versione completamente nuova, "
                            "con un approccio o uno stile diverso, ma basandoti sullo STESSO materiale."
                )],
                "final_article": ""
            }
        )

    elif action == "nuova_ricerca":
        feedback = human_response.get("feedback", "Trova nuove fonti e approfondisci l'argomento.")
        print("🔎 [Human Review] Nuova ricerca richiesta. Ritorno al Reasoner Subgraph...")
        return Command(
            goto="reasoner_subgraph",
            update={
                # Aggiorniamo la direttiva al reasoner in modo che il planner sappia cosa cercare di nuovo
                # decidere se svuotare il research_material precedente o se il reasoner ci aggiungerà info
                "prompt_to_reasoner": f"ATTENZIONE (Ricerca aggiuntiva richiesta): {feedback}",
                "final_article": "",
            }
        )

    else:
        print("🚫 [Human Review] Articolo annullato o risposta non riconosciuta.")
        return Command(goto=END)

async def save_article_node(state: BlogState) -> Command:
    """
    Si attiva solo dopo l'approvazione. 
    Estrae metadati, calcola embeddings e salva nel Knowledge Graph tramite MCP.
    """
    print("💾 [Save Node] Avvio estrazione metadati e salvataggio in locale...")
    
    final_article = state.get("final_article", "")
    intent = state.get("intent", "Unknown")
    subject = state.get("subject", "Unknown")
    specific_topic = state.get("specific_topic", "Unknown")
    current_suggestion_index = state.get("current_suggestion_index", 0)
    suggestions = state.get("suggestions", [])
    
    estrattore_metadati = writer_llm.with_structured_output(EstrazioneMetadatiArticolo, method="json_mode")
    
    try:
        system_prompt = get_metadata_extractor_prompt()
        messaggi_estrazione = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Estrai i metadati da questo testo in formato JSON:\n{final_article}")
        ]

        risultato_estrazione = await estrattore_metadati.ainvoke(messaggi_estrazione)
        
        concetti_collegati = risultato_estrazione.concetti_trovati
        fonti = risultato_estrazione.fonti
        relazioni_estratte = [rel.model_dump() for rel in risultato_estrazione.relazioni_concetti]
        claims_estratte = [claim.model_dump() for claim in risultato_estrazione.claims_estratti]
        
        prefisso = "Eserciziario" if intent.lower() == "eserciziario" else "ArticoloTeorico" if intent.lower() == "articoloteorico" else "TechNews"
        titolo_nodo = f"[{prefisso}] {specific_topic}"

        print("🧠 [Save Node] Calcolo dell'embedding per l'articolo completo...")
        testo_da_embeddare = f"Titolo: {titolo_nodo}\n\n"
        
        risultato_embedding = embedder.models.embed_content(
            model="gemini-embedding-2",
            contents=testo_da_embeddare
        )
        vettore_articolo = risultato_embedding.embeddings[0].values

        print(f"🔗 [Save Node] Connessione a MCP per salvare '{titolo_nodo}'...")
        
        risultato_salvataggio = await call_mcp_tool(
            tool_name="inserisci_articolo_agente",
            arguments={
                "titolo": titolo_nodo,
                "contenuto": final_article,
                "concetti_spiegati": concetti_collegati,
                "relazioni_concetti": relazioni_estratte,
                "materia": subject,
                "vettore": vettore_articolo,
                "fonti": fonti,
                "claims_articolo": claims_estratte 
            }
        )
        print(f"✅ [Save Node] Successo: {risultato_salvataggio}")

    except Exception as e:
        print(f"⚠️ [Save Node] Impossibile estrarre o salvare nel grafo: {str(e)}")

    
    # Se siamo in modalità "Planner" e ci sono ancora suggerimenti da esaminare
    if suggestions and (current_suggestion_index + 1) < len(suggestions):
        print(f"🔄 [Post-Save] Articolo salvato con successo. Passiamo al suggerimento successivo...")
        return Command(
            goto="hitl_planner",
            update={
                "current_suggestion_index": current_suggestion_index + 1,
            }
        )
    else:
        print("🏁 [Post-Save] Nessun altro suggerimento rimasto o flusso singolo completato. Terminazione.")
        return Command(goto=END)




#utils
def parse_human_response(raw: any) -> dict:
    """
    Normalizza qualsiasi risposta umana in un dict con chiave 'tipo'.
    Gestisce: lista, stringa JSON, stringa libera, dict, tipo sconosciuto.
    Supporta: approva, modifica, rigenera_testo, nuova_ricerca, annulla.
    """
    # Spacchetta lista se l'input arriva dentro un array
    if isinstance(raw, list) and len(raw) > 0:
        raw = raw[0]

    # È già un dict (es. inviato via JSON strutturato dalla UI) — caso ideale
    if isinstance(raw, dict):
        return raw

    # È una stringa — proviamo prima a decodificarla come JSON, poi passiamo all'analisi testuale
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Normalizzazione testo libero per il parsing delle parole chiave
        text = raw.lower().strip()
        
        # 1. OPZIONE: NUOVA RICERCA (Controlliamo prima le frasi composte)
        if any(k in text for k in ["nuova ricerca", "rifai ricerca", "cerca ancora", "nuove fonti", "new search", "approfondisci"]):
            return {"tipo": "nuova_ricerca", "feedback": raw}
        
        # 2. OPZIONE: RIGENERA SOLO IL TESTO (Stesso materiale)
        if any(k in text for k in ["rigenera", "riscrivi", "rifai testo", "regenerate", "cambia stile"]):
            return {"tipo": "rigenera_testo"}
        
        # 3. OPZIONE: APPROVA
        if any(k in text for k in ["approva", "approv", "ok", "sì", "si", "yes"]):
            return {"tipo": "approva"}
        
        # 4. OPZIONE: MODIFICA STRUTTURALE (Feedback generico sul testo esistente)
        if any(k in text for k in ["modifica", "cambia", "rivedi", "no", "feedback"]):
            return {"tipo": "modifica", "feedback": raw}

    # Paracadute finale se l'input non è interpretabile (es. l'utente chiude la finestra o scrive "stop")
    return {"tipo": "annulla"}