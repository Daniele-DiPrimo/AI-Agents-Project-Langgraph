import os
from dotenv import load_dotenv
from google import genai
from mcp import ClientSession, StdioServerParameters, stdio_client

# Carica le variabili d'ambiente dal file .env nella cartella del server
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.structures import ClassificationSchema, EstrazioneMetadatiArticolo
from src.state import BlogState

import json
from langgraph.types import interrupt, Command
from langgraph.graph import END

# --- 1. SETUP MODELLI GROQ ---
classifier_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
embedder = genai.Client()

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
    - macro_domain: La materia generale deve essere ESCLUSIVAMENTE una delle seguenti ["Algebra Lineare e Geometria", "Analisi Matematica I", "Database", "Economia Applicata Ingegneria", "Fisica I", "Fondamenti di Programmazione", "Analisi Matematica II", "Elettrotecnica", "Fisica II", "Internet e Sicurezza", "Machine Learning", "Programmazione Orientata agli Oggetti", "Sistemi Operativi", "Teoria dei Segnali", "Automatica", "Computer Architectures", "Comunicazioni Digitali", "Elettronica", "Software Design and Web Programming"].
    - specific_topic: L'argomento preciso. Estrailo direttamente dal prompt.
    - prompt_to_reasoner: Trasforma la richiesta dell'utente in una direttiva chiara per il nodo successivo che scriverà l'articolo.."
    - IMPORTANTE: Rimuovi QUALSIASI apostrofo, virgoletta o carattere speciale (es. $, '\', /) dai valori che generi nel JSON. Usa solo lettere e spazi.
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
async def writer_node(state: BlogState) -> dict:
    """
    Genera l'articolo finale o gli esercizi in Markdown, includendo obbligatoriamente le citazioni.
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
        
        REGOLE TASSATIVE (GROUNDING & CITATIONS):
        1. Basati ESCLUSIVAMENTE sul materiale di ricerca validato fornito qui sotto.
        2. NON inventare formule, sintassi o concetti non presenti in questa sintesi.
        3. Per OGNI formula, dato numerico o concetto tecnico utilizzato nell'esercizio o nella soluzione, DEVI inserire una citazione esplicita alla fonte.
        4. Il formato della citazione deve essere tra parentesi quadre con il nome esatto della fonte. Esempio: "Applicando il Teorema di Norton [Circuiti_Cap4.pdf]..."
        
        REGOLE DI FORMATTAZIONE:
        1. Crea 2 o 3 esercizi di difficoltà crescente (indica il livello: Base / Intermedio / Avanzato).
        2. Per ogni esercizio scrivi chiaramente la TRACCIA.
        3. Usa dati numerici e scenari realistici presi dal materiale.
        4. Fornisci la SOLUZIONE DETTAGLIATA per ogni esercizio con spiegazione dei passaggi, citando le fonti usate nei passaggi chiave.
        5. Usa Markdown per separare nettamente tracce e soluzioni.
        6. Scrivi in ITALIANO.
        """
    
    else:
        print(f"📝 -> Modalità: Redattore (Tipo: {intent})")
        sys_prompt = f"""Sei l'autore principale di un blog tecnico universitario.
        Scrivi un articolo di tipo '{intent}' sulla materia '{macro_domain}' sull'argomento '{specific_topic}'.
        
        REGOLE TASSATIVE (GROUNDING & CITATIONS):
        1. Basati ESCLUSIVAMENTE sul materiale di ricerca validato fornito qui sotto.
        2. NON inventare informazioni, concetti o codice non presenti in questa sintesi.
        3. Per OGNI affermazione tecnica, fatto o tesi che scrivi, DEVI inserire una citazione esplicita alla fonte direttamente nel testo.
        4. Il formato della citazione deve essere tra parentesi quadre con il nome esatto della fonte. Esempio: "La complessità del QuickSort nel caso pessimo è O(n^2) [Algoritmi_Cap3.pdf]."
        5. Se combini informazioni da più fonti, citale entrambe: [File1.pdf, File2.pdf, www.example.com].
        
        REGOLE DI FORMATTAZIONE:
        - Scrivi in ITALIANO con Markdown pulito.
        - Inizia con un titolo H1 chiaro e descrittivo.
        - Usa sezioni ben divise con H2 e H3.
        - Includi una sezione finale "## Fonti" elencando le fonti citate nel testo.
        """

    research_material_msg = HumanMessage(content=f"Materiale di riferimento per scrivere (con i nomi delle fonti indicati):\n{research_material}")

    # 3. Assemblaggio e invocazione
    llm_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Istruzioni specifiche per la redazione:\n{prompt_to_reasoner}"),
        research_material_msg
    ]

    final_draft = await writer_llm.ainvoke(llm_messages)
    testo_generato = final_draft.content
    print("✅ [Writer] Stesura completata.")

    # 4. Ritorno dello stato aggiornato
    return {"final_article": testo_generato}

# =====================================================
# FINE NODI DI SCRITTURA (HELPER_WRITER - WRITER - EXERCISES WRITER)
# =====================================================

# =====================================================
# INIZIO NODO HITL (HUMAN_REVIEW --> NODO / _PARSE_HUMAN_RESPONSE_ PER GESTIRE LA RISPOSTA)
# =====================================================

async def human_review_node(state: BlogState) -> Command:
    """
    HITL: mette in pausa il grafo per la revisione umana dell'articolo.
    Legge: final_article, intent, macro_domain, specific_topic
    Azioni possibili: approva (SALVA NEL GRAFO) → END | modifica → writer | annulla → END
    """

    final_article = state.get("final_article", "")
    intent = state.get("intent", "Unknown")
    macro_domain = state.get("macro_domain", "Unknown")
    specific_topic = state.get("specific_topic", "Unknown")

    if not final_article:
        print("⚠️ [Human Review] final_article vuoto, approvazione automatica senza salvataggio.")
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
        print("✅ [Human Review] Articolo approvato. Avvio salvataggio nel Knowledge Graph...")
        
        # ==========================================================
        # MEMORY LOOP (Eseguito SOLO se approvato)
        # ==========================================================
        estrattore_metadati = writer_llm.with_structured_output(EstrazioneMetadatiArticolo, method="json_mode")
        
        try:
            # 1. Estrazione concetti e fonti con Prompt potenziato
            schema_richiesto = """
            {
              "concetti_trovati": ["concetto1", "concetto2"],
              "relazioni_concetti": [
                {
                  "origine": "concetto1", 
                  "tipo_relazione": "SI_BASA_SU", 
                  "destinazione": "concetto2", 
                  "dettaglio": "breve spiegazione"
                }
              ],
              "fonti_documentali": ["file1.pdf", "file2.pdf"],
              "link_esterni": ["https://esempio.com"],
              "claims_estratti": [
                {
                  "affermazione": "Il concetto1 riduce i tempi di latenza",
                  "concetto_riferimento": "concetto1"
                }
              ]
            }
            """
            
            messaggi_estrazione = [
                SystemMessage(
                    content=(
                        "Sei un estrattore dati specializzato. Analizza l'articolo fornito. "
                        "DEVI rispondere ESCLUSIVAMENTE con un oggetto JSON valido. "
                        "Il tuo JSON DEVE usare ESATTAMENTE queste chiavi e rispettare questa struttura:\n"
                        f"{schema_richiesto}\n"
                        "REGOLE:\n"
                        "1. Se non ci sono file PDF, link o claims, restituisci array vuoti [].\n"
                        "2. In 'relazioni_concetti', usa SOLO: SI_BASA_SU, È_UN_TIPO_DI, COMPOSTO_DA, RISOLVE_USA."
                    )
                ),
                HumanMessage(content=f"Estrai i metadati da questo testo in formato JSON:\n{final_article}")
            ]

            risultato_estrazione = await estrattore_metadati.ainvoke(messaggi_estrazione)
            
            concetti_collegati = risultato_estrazione.concetti_trovati
            documenti_usati = risultato_estrazione.fonti_documentali
            link_trovati = risultato_estrazione.link_esterni

            # Convertiamo le relazioni in una lista di dizionari per inviarli via MCP
            relazioni_estratte = [rel.model_dump() for rel in risultato_estrazione.relazioni_concetti]

            claims_estratte = [claim.model_dump() for claim in risultato_estrazione.claims_estratti] # <-- NUOVO
            
            # Generazione del titolo per Neo4j
            prefisso = "Esercizi" if intent.lower() == "esercizio" else "Blog"
            titolo_nodo = f"[{prefisso}] {specific_topic}"

            # 2. CALCOLO DELL'EMBEDDING DELL'ARTICOLO (NUOVO)
            print("🧠 Calcolo dell'embedding per l'articolo completo...")
            # Combiniamo titolo e testo. Limitiamo a ~8000 caratteri in caso di articoli enormi per evitare limiti di token
            testo_da_embeddare = f"Titolo: {titolo_nodo}\n\nContenuto: {final_article[:8000]}"
            
            # NOTA: Assicurati che 'embedder' sia definito globalmente in questo file come 'embedder = genai.Client()'
            risultato_embedding = embedder.models.embed_content(
                model="gemini-embedding-2",
                contents=testo_da_embeddare
            )
            vettore_articolo = risultato_embedding.embeddings[0].values

            # 3. Salvataggio via MCP
            print(f"💾 [Human Review] Connessione a MCP per salvare '{titolo_nodo}' con relazioni e fonti...")
            
            percorso_mcp = os.getenv("MCP_SERVER_URI", "")
            server_params = StdioServerParameters(command="python", args=[percorso_mcp])
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    risultato_salvataggio = await session.call_tool("inserisci_articolo_agente", arguments={
                        "titolo": titolo_nodo,
                        "contenuto": final_article,
                        "concetti_spiegati": concetti_collegati,
                        "relazioni_concetti": relazioni_estratte, # <-- NUOVO PARAMETRO
                        "materia": macro_domain,
                        "vettore": vettore_articolo, # <--- PASSIAMO IL VETTORE QUI
                        "fonti_documentali": documenti_usati, # <-- NUOVO
                        "link_esterni": link_trovati,          # <-- NUOVO
                        "claims_articolo": claims_estratte # <-- NUOVO PARAMETRO
                    })
                    print(f"✅ [Review-MCP]: {risultato_salvataggio.content[0].text}")

        except Exception as e:
            print(f"⚠️ [Review-MCP] Impossibile salvare nel grafo o calcolare l'embedding: {str(e)}")

        # Esce dal grafo dopo aver salvato
        return Command(goto=END)

    elif action == "modifica":
        feedback = human_response.get(
            "feedback",
            "Revisione generale richiesta senza dettagli specifici."
        )
        print(f"🔄 [Human Review] Modifica richiesta: {feedback[:100]}...")

        return Command(
            goto="writer",
            update={
                "messages": [HumanMessage(
                    content=f"[FEEDBACK REVISORE UMANO] {feedback}. "
                            f"Riscrivi l'articolo tenendo conto di questo feedback. "
                            f"Il materiale di ricerca validato è già disponibile in research_material."
                )],
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