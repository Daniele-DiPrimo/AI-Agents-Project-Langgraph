from langchain_groq import ChatGroq
from langchain_core.messages import  SystemMessage, HumanMessage
from src.structures import  FullSourcesEvaluationSchema, CompletenessEvaluationSchema
from src.state import  ReasonerState
from src.tools import blog_tools
from langgraph.graph import END
import re


planner_completeness_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
source_evaluator_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

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
    prompt_to_reasoner = state.get("prompt_to_reasoner", "NotSpecified")

    missing_info = state.get("missing_info", "Questa è la prima ricerca. Inizia esplorando l'argomento.")
 
    planner_llm_w_tools = planner_completeness_llm.bind_tools(
        blog_tools,
        tool_choice="auto",
        parallel_tool_calls=False
    )
 
    sys_prompt = f"""Sei l'Agente Ricercatore Capo. Il tuo unico scopo è pianificare ed eseguire la ricerca di informazioni chiamando i tool appropriati.
 
    CONTESTO DELLA RICERCA:
    - Intent: {intent}
    - Dominio: {macro_domain}
    - Topic specifico: {specific_topic}

    OBIETTIVO FINALE DEL REDATTORE (Usa questo SOLO come contesto per capire cosa cercare):
    "{prompt_to_reasoner}"

    STATUS RICERCA ATTUALE (Feedback del Revisore):
    "{missing_info}"
    
    REGOLE DI SELEZIONE DEI TOOL:
    - Usa 'tavily' per ricerche generiche, notizie recenti (News) o concetti di base.
    - Usa 'semantic_scholar' per ricerche accademiche, paper e teoria approfondita.
    
    REGOLE OPERATIVE RIGOROSE:
    1. Usa l'OBIETTIVO FINALE per capire il livello di dettaglio e il taglio che dovrà avere l'articolo, formulando query di ricerca precise.
    2. Ti è ASSOLUTAMENTE VIETATO scrivere l'articolo o rispondere alla direttiva dell'Obiettivo Finale. Devi SOLO cercare e inoltrare i risultati grezzi.
    3. Se ritieni che le informazioni presenti nella cronologia siano sufficienti per coprire l'Obiettivo Finale, fermati e rispondi senza invocare ulteriori tool.
    """
 
    messages = [SystemMessage(content=sys_prompt)]
    answer = planner_llm_w_tools.invoke(messages)
 
    return {"tool_plan": answer.tool_calls}


def tool_executor_node(state: ReasonerState) -> dict:
    """Esegue le chiamate generate da bind_tools e salva in raw_results."""
    print("🛠️ [Tool Executor] Esecuzione ricerche...")
    
    tool_plan = state.get("tool_plan", [])
    raw_results = []
    visited_urls = state.get("visited_urls", [])
    
    # Creiamo un dizionario veloce per richiamare i tuoi tool reali tramite il loro nome
    tools_by_name = {tool.name: tool for tool in blog_tools}

    # Iteriamo sulle chiamate native estratte nel Planner
    for tool_call in tool_plan:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] # È già un dizionario con gli argomenti (es. {"query": "..."})
        
        if tool_name in tools_by_name:
            try:
                # Eseguiamo il tool passando gli argomenti scompattati
                result = tools_by_name[tool_name].invoke(tool_args)
                result_str = str(result)
                
                extracted_links = re.findall(r'https?://[^\s"\'\},\]]+', result_str)
                
                for link in extracted_links:
                    if link not in visited_urls:
                        visited_urls.append(link)

                # Salviamo il risultato puro e crudo nel nostro campo specifico
                raw_results.append({
                    "tool_used": tool_name,
                    "query_used": str(tool_args),
                    "raw_output": str(result)
                })
                print(f"   -> Tool '{tool_name}' eseguito con successo.")
            except Exception as e:
                print(f"   -> ⚠️ Errore nell'esecuzione del tool {tool_name}: {e}")

    # Aggiorniamo ESCLUSIVAMENTE raw_results
    return {"raw_results": raw_results,
            "visited_urls": visited_urls}
    

# 2. IL NODO AGGIORNATO
def source_evaluator_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile ESCLUSIVAMENTE della valutazione delle fonti.
    """
    print("🔍 [Source Evaluator] Valutazione granulare delle fonti...")

    structured_source_evaluator_llm = source_evaluator_llm.with_structured_output(
        FullSourcesEvaluationSchema,
        method="json_mode"
    )

    sys_prompt = """Sei un revisore accademico spietato.
    Analizza i risultati delle ricerche. Per ogni fonte valuta da 0.0 a 1.0:
    1. 'source_reliability': L'affidabilità della fonte.
       - 0.9/1.0 = Paper accademici, documentazione ufficiale, blog tecnici riconosciuti.
       - 0.6/0.8 = Articoli divulgativi validi ma generalisti.
       - < 0.5 = Forum non verificati, spam, fonti dubbie.
    2. 'information_relevance': L'attinenza al topic richiesto.
       - 0.9/1.0 = Contiene dati tecnici, codice o definizioni esatte richieste.
       - 0.5/0.8 = Parla dell'argomento ma in modo superficiale.
       - < 0.5 = Fuori tema o menziona l'argomento solo di sfuggita.
    
    Devi essere severo. Se la fonte è generica, penalizzala.
    
    Usa ESATTAMENTE questa struttura JSON:
    {
    "judgments": [
        {
        "index_source": 0,
        "source_reliability": 0.0,
        "information_relevance": 0.0,
        "reasoning": "Spiega brevemente i due punteggi assegnati"
        }
    ],
    "need_new_search": false
    }"""
    
    raw_results = state.get("raw_results", [])
    if not raw_results:
        return {"sources_evaluated": False, "approved_sources": state.get("approved_sources", [])}

    results_text = "\n\n".join([
        f"--- ID FONTE: {i} ---\nTOOL: {res['tool_used']} | QUERY: {res['query_used']}\nOUTPUT: {res['raw_output']}\n-------------------" 
        for i, res in enumerate(raw_results)
    ])
    
    llm_input = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"RISULTATI GREZZI DA VALUTARE:\n{results_text}")
    ]

    judgment: FullSourcesEvaluationSchema = structured_source_evaluator_llm.invoke(llm_input)

    THRESHOLD_RELIABILITY = 0.70
    THRESHOLD_RELEVANCE = 0.70
    
    approved_sources = []
    not_approved_sources = []
    
    for v in judgment.judgments:
        # LOGICA AND: Deve superare ENTRAMBE le soglie
        if v.source_reliability >= THRESHOLD_RELIABILITY and v.information_relevance >= THRESHOLD_RELEVANCE:
            idx = v.index_source
            whole_content = "Testo originale non trovato."
            id_label = "Fonte Sconosciuta" 
            
            if 0 <= idx < len(raw_results):
                whole_content = raw_results[idx]['raw_output']
                id_label = f"Ricerca {idx} tramite {raw_results[idx]['tool_used']}"
            
            approved_sources.append({
                "id_source": id_label, 
                "source_reliability": v.source_reliability,
                "information_relevance": v.information_relevance,
                "overall_score": (v.source_reliability + v.information_relevance) / 2.0, 
                "reasoning": v.reasoning, 
                "content": whole_content 
            })
        else:
            not_approved_sources.append(v)

    print(f"   -> Fonti analizzate: {len(judgment.judgments)} | "
          f"Approvate: {len(approved_sources)} | "
          f"Bocciate: {len(not_approved_sources)}")

    if judgment.need_new_search or len(approved_sources) == 0:
        return {
            "approved_sources": state.get("approved_sources", []), 
            "sources_evaluated": False
        }
    else:
        existing = state.get("approved_sources", [])
        return {
            "approved_sources": existing + approved_sources,
            "sources_evaluated": True
        }
# ==========================================
# COMPLETENESS EVALUATOR
# ==========================================
def completeness_evaluator_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile della valutazione della completezza complessiva del materiale.
    """
    print("⚖️ [Completeness Evaluator] Controllo completezza materiale...")
    completeness_structured_llm = planner_completeness_llm.with_structured_output(
        CompletenessEvaluationSchema
    )

    intent         = state.get("intent", "NotSpecified")
    macro_domain   = state.get("macro_domain", "NotSpecified")
    specific_topic = state.get("specific_topic", "NotSpecified")
    approved_sources = state.get("approved_sources", [])
    sources_evaluated = state.get("sources_evaluated", False)
    iterations     = state.get("iterations", 0)

    if not sources_evaluated:
        return {
            "is_complete": False,
            "iterations": iterations + 1,
            "missing_info": "⚠️ Tutte le fonti recenti sono state scartate. Usa parole chiave o tool diversi per la prossima ricerca."
        }

    sys_prompt = f"""Sei il Chief Editor accademico di un blog universitario tecnico.
    Il tuo compito è valutare con estremo rigore se il materiale raccolto finora è sufficientemente profondo e completo per scrivere un contenuto di livello universitario.
    
    OBIETTIVO: articolo di tipo [{intent}], materia [{macro_domain}], argomento [{specific_topic}].
    
    CRITERI DI SUFFICIENZA (Devono essere tutti soddisfatti):
    1. Profondità tecnica: Ci sono dettagli tecnici, architettonici o matematici reali, non solo definizioni da dizionario?
    2. Completezza: Le sfaccettature principali dell'argomento sono coperte?
    3. Praticità: È presente almeno un caso d'uso, un esempio pratico o del codice (se pertinente all'argomento)?
    
    REGOLA FONDAMENTALE (STRICT GROUNDING):
    Il Writer finale non potrà inventare nulla. Se un dettaglio manca nel materiale estratto qui sotto, mancherà anche nell'articolo finale. Se ritieni che l'articolo finale risulterebbe troppo superficiale basandosi solo su questi testi, DEVI bocciare la completezza.
    
    REGOLE OPERATIVE:
    - NON fare domande all'utente.
    - Se l'informazione è insufficiente, metti "is_complete": false e in 'missing_info' scrivi 3-4 parole chiave mirate in inglese per guidare la prossima ricerca del Planner verso i concetti tecnici mancanti."""

    # AGGIORNATO: Mostriamo entrambi i punteggi al Revisore
    sources_summary = "\n".join([
        f"- {s.get('id_source', 'N/A')} (Affidabilità: {s.get('source_reliability', 0)} | Attinenza: {s.get('information_relevance', 0)}): {s.get('reasoning', '')}\n  Testo Originale: {s.get('content', '')}"
        for s in approved_sources
    ])
    human_msg = HumanMessage(content=f"Materiale raccolto finora:\n{sources_summary}")

    answer: CompletenessEvaluationSchema = completeness_structured_llm.invoke([
        SystemMessage(content=sys_prompt),
        human_msg
    ])

    if answer.is_complete:
        research_material = f"# Materiale di ricerca validato\n\n"
        research_material += f"**Tipo articolo:** {intent} | **Dominio:** {macro_domain} | **Topic:** {specific_topic}\n\n"
        
        for s in approved_sources:
            # AGGIORNATO: Mostriamo entrambi i punteggi nel Markdown per il Writer
            research_material += f"## {s.get('id_source', 'Sconosciuta')} (Affidabilità: {s.get('source_reliability', 0.0):.1f} | Attinenza: {s.get('information_relevance', 0.0):.1f})\n"
            research_material += f"**Motivazione:** {s.get('reasoning', '')}\n\n"
            research_material += f"**Testo grezzo dal web:**\n> {s.get('content', 'Nessun testo estratto.')}\n\n"
            research_material += "---\n\n"

        print(f"✅ [Coverage Evaluator] Materiale sufficiente. Fonti approvate in totale: {len(approved_sources)}")
        return {
            "is_complete": True,
            "research_material": research_material,
            "iterations": iterations + 1
        }
    else:
        print(f"🔄 [Coverage Evaluator] Materiale incompleto. Manca: {answer.missing_info}")
        return {
            "is_complete": False,
            "iterations": iterations + 1,
            "missing_info": f"INFO MANCANTI: Il revisore ha detto che manca: '{answer.missing_info}'. Fai un'altra ricerca mirata su questo."
        }

# =====================================================
# FINE NOODI DEL SOTTOGRAFO DI REASONING [PLANNER, SOURCE_EVALUATOR, COMPLETENESS_EVALUATOR]
# =====================================================