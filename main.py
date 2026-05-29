from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# IMPORTANTE: Importiamo il builder, NON il grafo già compilato
from src.graph import builder 

def run_blog_system(prompt: str):
    print(f"\n🚀 Avvio elaborazione per: '{prompt}'\n")
    print("-" * 50)
    
    # 1. Inizializziamo il Checkpointer per questa esecuzione locale
    memory = MemorySaver()
    
    # 2. Compiliamo il grafo aggiungendo memoria e punto di interruzione
    blog_system_local = builder.compile(
        checkpointer=memory,
        interrupt_after=["writer"]
    )
    
    # 3. Definiamo il Thread ID (Obbligatorio quando si usa la memoria)
    config = {"configurable": {"thread_id": "sessione_terminale_01"}}
    
    initial_state = {
        "messages": [HumanMessage(content=prompt)]
    }
    
    # --- FASE DI ESECUZIONE E LOG RADIOGRAFICO ---
    for event in blog_system_local.stream(initial_state, config=config, stream_mode="updates"):
        
        for node_name, node_state in event.items():
            print(f"\n\n{'='*20} 🟢 OUTPUT DAL NODO: {node_name.upper()} {'='*20}")
            
            # Se siamo nel classificatore, stampiamo le sue variabili estratte
            if node_name == "classifier":
                print(f"📌 Intento: {node_state.get('intent')}")
                print(f"📌 Dominio: {node_state.get('macro_domain')}")
                print(f"📌 Topic Specifico: {node_state.get('specific_topic')}")
            
            # Se ci sono messaggi, li stampiamo formattati
            # Se ci sono messaggi, li analizziamo
            elif "messages" in node_state:
                ultimo_messaggio = node_state["messages"][-1]
                
                # Se il messaggio contiene chiamate ai tool, estraiamo la nostra giustificazione!
                if hasattr(ultimo_messaggio, "tool_calls") and ultimo_messaggio.tool_calls:
                    for call in ultimo_messaggio.tool_calls:
                        nome_tool = call["name"]
                        argomenti = call["args"]
                        
                        # Estraiamo i campi
                        motivo = argomenti.get("giustificazione", "Nessuna giustificazione fornita")
                        query = argomenti.get("query", "")
                        
                        print(f"\n🧠 [PENSIERO DELL'AGENTE]: {motivo}")
                        print(f"🛠️ [AZIONE]: Chiama il tool '{nome_tool}' con query: '{query}'")
                        
                else:
                    # Stampa classica di fallback
                    ultimo_messaggio.pretty_print()

    # --- FASE HUMAN-IN-THE-LOOP (HITL) ---
    snapshot = blog_system_local.get_state(config)
    
    # LangGraph salva lo stato di interruzione anche nei metadati. 
    # Visto che writer va a END, verifichiamo semplicemente se l'esecuzione ha finito di ciclare
    print("\n" + "⚠️ " * 15)
    print("PAUSA HITL: L'agente ha finito di scrivere. Leggi l'articolo qui sopra.")
    print("⚠️ " * 15)
    
    scelta = input("\nPremi [INVIO] per chiudere con successo, oppure [NO] per rifiutarlo: ")
    
    if scelta.strip() == "":
         print("\n✅ Articolo confermato! Processo completato.")
    else:
         print("\n❌ Articolo rifiutato. (Usa LangGraph Studio per ricaricare il nodo e fornire feedback).")

if __name__ == "__main__":
    test_prompt = "Scrivi un articolo di Teoria sulle liste in Rust"
    run_blog_system(test_prompt)