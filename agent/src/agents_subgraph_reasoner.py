from langchain_groq import ChatGroq
from langchain_core.messages import  SystemMessage, HumanMessage

from src.structures import CompletenessEvaluationSchema, SourceEvaluationSchema

from src.state import  ReasonerState

from src.tools import blog_tools

from src.prompts import (
    get_reasoner_prompt,
    get_source_evaluator_prompt,
    get_completeness_evaluator_prompt,
)

reasoner_completeness_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
source_evaluator_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)



def reasoner_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile del piano esecutivo per la ricerca.
    Decide quali tool chiamare e genera le query.
    Scrive solo in: messages (loop ReAct interno).
    """
 
    intent = state.get("intent", "NotSpecified")
    subject  = state.get("subject", "NotSpecified")
    specific_topic = state.get("specific_topic", "NotSpecified")
    prompt_to_reasoner = state.get("prompt_to_reasoner", "NotSpecified")

    missing_info = state.get("missing_info", "Questa è la prima ricerca. Inizia esplorando l'argomento.")

    # Genera il prompt dinamicamente
    sys_prompt = get_reasoner_prompt(
        intent, subject, specific_topic, 
        prompt_to_reasoner, missing_info
    )

    reasoner_llm_w_tools = reasoner_completeness_llm.bind_tools(
        blog_tools,
        tool_choice="any",
        parallel_tool_calls=False
    )

    answer = reasoner_llm_w_tools.invoke([SystemMessage(content=sys_prompt)])

    print(f"📝 [Reasoner] Piano generato: {answer.tool_calls}")
 
    return {"tool_plan": answer.tool_calls}


async def tool_executor_node(state: ReasonerState) -> dict:
    """Esegue le chiamate generate da bind_tools e salva in raw_results."""
    print("🛠️ [Tool Executor] Esecuzione ricerche...")
    
    tool_plan = state.get("tool_plan", [])
    raw_results = []
    
    # Creiamo un dizionario veloce per richiamare i tuoi tool reali tramite il loro nome
    tools_by_name = {tool.name: tool for tool in (blog_tools)}

    tool_call = tool_plan[-1]
    
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
        
    try:
        # Eseguiamo il tool passando gli argomenti scompattati
        result = await tools_by_name[tool_name].ainvoke(tool_args)

        raw_results = result
            
        print(f"   -> Tool '{tool_name}' eseguito con successo.")
    except Exception as e:
        print(f"   -> ⚠️ Errore nell'esecuzione del tool {tool_name}: {e}")

    return {
        "raw_results": raw_results,
        "tool_used": tool_name,
        "tool_args": tool_args
    }



def source_evaluator_node(state: ReasonerState) -> dict:
    print("🔍 [Source Evaluator] Valutazione batch delle fonti...")

    structured_source_evaluator_llm = source_evaluator_llm.with_structured_output(SourceEvaluationSchema)

    # Estrazione stato
    raw_results = state.get("raw_results", [])
    prompt_to_reasoner = state.get("prompt_to_reasoner", "")
    graph_results = state.get("graph_results", [])
    iterations = state.get("iterations", 0)

    # 1. Inizializziamo le liste GLOBALI fuori dal ciclo per non sovrascriverle
    approved_sources = []
    not_approved_sources = []
    
    THRESHOLD_RELEVANCE = 0.70
    THRESHOLD_RELIABILITY = 0.70

    # CORREZIONE 1: Usiamo le parentesi quadre per creare una Lista, non un Set
    target_results = [graph_results, raw_results] if iterations == 0 else [raw_results]

    for r in target_results:
        # Usiamo .get() per sicurezza, nel caso la chiave manchi
        results = r.get("results", []) 
        
        # CORREZIONE 2: Uscita anticipata per il SINGOLO giro, non per tutta la funzione
        if not results:
            continue

        # 2. Prepara il testo delle fonti per il prompt
        sources_text = f"QUERY: {prompt_to_reasoner}\n\n"
        for item in results:
            source = item.get("source", item.get("url", "Fonte Mancante"))
            content = item.get("content", "Contenuto Mancante")
            if r is graph_results:
                id = item.get("id")
                sources_text += (f"--- FONTE ---\nFonte: {source}\n ID {id} \n Content: {content}\n\n")
            else: 
                sources_text += (f"--- FONTE ---\nFonte: {source}\n  \n Content: {content}\n\n")

        # 3. Chiamata all'LLM
        llm_input = [
            SystemMessage(content=get_source_evaluator_prompt()),
            HumanMessage(content=f"Valuta le seguenti fonti:\n{sources_text}")
        ]
        judgment_batch = structured_source_evaluator_llm.invoke(llm_input)

        # 4. Elaborazione dei giudizi
        results_by_source = {item.get("id", item.get("source")): item for item in results} 

        for j in judgment_batch.judgments:
            
            source_data = {
                "source_reliability": j.source_reliability,
                "source_relevance": j.source_relevance,
                "overall_score": (j.source_reliability + j.source_relevance) / 2.0, 
                "reasoning": j.reasoning, 
                "content": results_by_source.get(j.id, results_by_source.get(j.source)).get("content", ""),
                "source": j.source
            }

            # 5. Logica di approvazione condizionale
            # Usiamo 'is' invece di '==' per verificare l'esatta identità dell'oggetto in memoria
            if r is graph_results: # Ricerca da K-RAG (Interna)
                is_approved = j.source_relevance >= THRESHOLD_RELEVANCE
            else: # Ricerca Web (Esterna)
                is_approved = (j.source_reliability >= THRESHOLD_RELIABILITY and 
                               j.source_relevance >= THRESHOLD_RELEVANCE)

            # Aggiungiamo i risultati alle liste globali
            if is_approved:
                approved_sources.append(source_data)
            else:
                not_approved_sources.append(source_data)

    # 6. Ritorno finale solo quando il ciclo ha processato tutte le fonti necessarie
    return {
        "approved_sources": approved_sources,
        "not_approved_sources": not_approved_sources,
        "sources_evaluated": len(approved_sources) > 0
    }



def completeness_evaluator_node(state: ReasonerState) -> dict:
    """
    Nodo responsabile della valutazione della completezza complessiva del materiale.
    """
    print("⚖️ [Completeness Evaluator] Controllo completezza materiale...")

    MAX_ITERATIONS = 5

    completeness_structured_llm = reasoner_completeness_llm.with_structured_output(CompletenessEvaluationSchema)

    intent         = state.get("intent", "NotSpecified")
    subject   = state.get("subject", "NotSpecified")
    specific_topic = state.get("specific_topic", "NotSpecified")
    approved_sources = state.get("approved_sources", [])
    iterations     = state.get("iterations", 0)
    sources_evaluated = state.get("sources_evaluated", False)

    sys_prompt = get_completeness_evaluator_prompt(intent, subject, specific_topic)

    # 1. Gestione uscita anticipata se mancano fonti valutate
    if not sources_evaluated and iterations < MAX_ITERATIONS:
            return {
                "is_complete": False,
                "iterations": iterations + 1,

            }
    
    sources_summary = "\n".join([
        f"- {s.get('source', 'N/A')} (Affidabilità: {s.get('source_reliability', 0)} | Attinenza: {s.get('source_relevance', 0)}): {s.get('reasoning', '')}\n  Testo Originale: {s.get('content', '')}" 
        for s in approved_sources
    ])

    human_msg = HumanMessage(content=f"Materiale raccolto finora:\n{sources_summary}")

    answer = completeness_structured_llm.invoke([
        SystemMessage(content=sys_prompt),
        human_msg
    ])

    research_material = f"# Materiale di ricerca validato\n\n"
    research_material += f"**Tipo articolo:** {intent} | **Dominio:** {subject} | **Topic:** {specific_topic}\n"
    
    for s in approved_sources:
        research_material += f"## {s.get('source', 'Sconosciuta')} (Affidabilità: {s.get('source_reliability', 0.0):.1f} | Attinenza: {s.get('source_relevance', 0.0):.1f})\n"
        research_material += f"**Motivazione:** {s.get('reasoning', '')}\n\n"
        research_material += f"**Testo grezzo dal web:**\n> {s.get('content', 'Nessun testo estratto.')}\n\n"
        research_material += "---\n\n"

    is_complete = answer.is_complete or iterations >= MAX_ITERATIONS
  
    return {
        "is_complete": True if is_complete else False, 
        "research_material": research_material,
        "iterations": iterations + 1,
        "missing_info": ""  if is_complete else f"INFO MANCANTI: Il revisore ha detto che manca: '{answer.missing_info}'. Fai un'altra ricerca mirata su questo."
    }