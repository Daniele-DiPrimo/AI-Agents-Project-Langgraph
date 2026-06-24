import os
import chromadb 
from chromadb.utils import embedding_functions

try: 
    chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
    )
except Exception as e:
    print(f"Impossibile connettersi a chroma cloud. {e}")

google_ef = embedding_functions.GoogleGenaiEmbeddingFunction(
    model_name="gemini-embedding-2"
)

collection = chroma_client.get_or_create_collection(name="UniAgent_RAG", embedding_function=google_ef)

def rag_search(queries: list[str], subject: str, top_k: int=3) -> list[dict]:
    """
    Esegue query sequenziali su ChromaDB, aggirando il bug di batching 
    dell'Embedding di Google. Deduplica i chunk tramite ID e restituisce
    una lista di dizionari pulita.
    """
    chunk_unici = {}
    
    subject_variations = [
        subject,                  # Originale: "Teoria dei Segnali"
        subject.lower(),          # Tutto minuscolo: "teoria dei segnali"
        subject.upper(),          # Tutto maiuscolo: "TEORIA DEI SEGNALI"
        subject.title(),          # Iniziali maiuscole: "Teoria Dei Segnali"
        subject.capitalize()      # Solo la prima: "Teoria dei segnali"
    ]

    # Controllo di sicurezza rapido
    if not queries:
        return []
    
    # Cicliamo le query una ad una: 1 stringa -> 1 vettore -> 1 risultato. Infallibile.
    for q in queries:
        try:
            rag_results = collection.query(
                query_texts=[q], # Avvolgiamo la singola stringa in una lista
                n_results=top_k,
                where={"subject": {"$in": subject_variations}} # Barriera anti-allucinazioni attivata
            )

            # Controllo di sicurezza per la singola estrazione
            if rag_results and rag_results.get('documents') and rag_results['documents'][0]:
                
                documenti = rag_results['documents'][0]
                ids = rag_results['ids'][0]
                
                # Gestione sicura dei metadati
                if rag_results.get('metadatas') and rag_results['metadatas'][0]:
                    metadati = rag_results['metadatas'][0]
                else:
                    metadati = [{}] * len(documenti)
                
                # Zip e Deduplicazione Globale
                for doc_id, doc_text, meta in zip(ids, documenti, metadati):
                    if doc_id not in chunk_unici:
                        chunk_unici[doc_id] = {
                            "chunk_id": doc_id,
                            "testo": doc_text,
                            "fonte": meta.get("source", "Sconosciuta")
                        }
                        
        except Exception as e:
            print(f"🔥 Errore ChromaDB sulla query '{q}': {e}")
            
    return list(chunk_unici.values())

