import os
import asyncio
from pydantic import BaseModel, Field
from typing import List
import hashlib

# MCP Client Imports
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from google import genai

from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. CONFIGURAZIONE E MODELLI
# ==========================================
# 🚨 CAMBIA QUESTO PERCORSO: Inserisci il percorso assoluto del tuo server MCP
PERCORSO_SERVER_MCP = os.getenv("MCP_SERVER_URI", "")

MATERIE_CONSENTITE = [
    "Algebra Lineare e Geometria", "Analisi Matematica I", "Database", "Economia Applicata Ingegneria", "Fisica I", "Fondamenti di Programmazione",
    "Analisi Matematica II", "Elettrotecnica", "Fisica II", "Internet e Sicurezza", "Machine Learning", "Programmazione Orientata agli Oggetti", "Sistemi Operativi", "Teoria dei Segnali",
    "Automatica", "Computer Architectures", "Comunicazioni Digitali", "Elettronica", "Software Design and Web Programming"
]

class NodoEstratto(BaseModel):
    macro_categoria: str = Field(description="SOLO TRA: Concetto_Teorico, Componente_Tecnologico, Processo_Algoritmo, Persona")
    nome: str = Field(description="Nome univoco, es. 'Teorema di Shannon'")
    micro_categoria: str = Field(description="Sottocategoria libera")
    descrizione_breve: str = Field(description="Massimo 15 parole.")

class RelazioneEstratta(BaseModel):
    origine: str = Field(description="Nome del nodo di origine")
    tipo_relazione: str = Field(description="SOLO TRA: SI_BASA_SU, È_UN_TIPO_DI, COMPOSTO_DA, RISOLVE_USA")
    destinazione: str = Field(description="Nome del nodo di destinazione")
    dettaglio: str = Field(description="Contesto specifico in poche parole")

class ClaimEstratto(BaseModel):
    affermazione: str = Field(description="Una frase completa che esprime una tesi, una regola o un fatto chiave (max 15 parole).")
    concetto_riferimento: str = Field(description="Il nome esatto del nodo Concetto_Teorico estratto a cui si riferisce.")

class GrafoEstratto(BaseModel):
    nodi: List[NodoEstratto]
    relazioni: List[RelazioneEstratta]
    claims: List[ClaimEstratto]


# Inizializzazione Modelli AI (Globali per riutilizzo)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=4000)
estrattore = llm.with_structured_output(GrafoEstratto)
embedder = genai.Client()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ==========================================
# 2. FUNZIONI MODULARI DI INGESTIONE
# ==========================================

def calcola_hash_pdf(filepath: str) -> str:
    """Calcola l'impronta digitale unica (SHA-256) del file PDF."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

async def inizializza_ontologia(session: ClientSession):
    """Crea i nodi 'hub' per tutte le materie universitarie."""
    print("📚 Inizializzazione nodi Materia...")
    for materia in MATERIE_CONSENTITE:
        print(f"   -> Creazione nodo per: {materia}")
        try:
            risultato = await session.call_tool("crea_nodo", arguments={
                "macro_categoria": "Materia",
                "nome_entita": materia,
                "micro_categoria": "Corso_Triennale",
                "descrizione_breve": "Corso Ingegneria Informatica UNICT"
            })
            # Stampa la risposta del server (Successo o Errore DB)
            print(f"      Risposta MCP: {risultato.content[0].text}")
        except Exception as e:
            print(f"      ❌ Errore critico MCP: {str(e)}")

async def salva_chunk_e_vettore(session: ClientSession, chunk_id: str, testo_chunk: str, nome_file: str, nome_materia: str):
    """Calcola l'embedding di un chunk di testo e lo salva nel database via MCP, collegandolo al Documento genitore."""
    vettore = embedder.models.embed_content(
        model="gemini-embedding-2",
        contents=testo_chunk
    )

    vettore = vettore.embeddings[0].values
    
    # Salva il nodo chunk
    await session.call_tool("crea_nodo_chunk", arguments={
        "chunk_id": chunk_id,
        "testo": testo_chunk,
        "vettore": vettore,
        "sorgente": nome_file
    })
    
    # MODIFICA GERARCHICA: Il Chunk ora si lega al suo Documento (nome_file), NON più alla materia!
    await session.call_tool("crea_relazione", arguments={
        "entita_origine": chunk_id,
        "tipo_relazione": "APPARTIENE_A",
        "entita_destinazione": nome_file,
        "dettaglio": "Sezione del documento"
    })

async def estrai_e_salva_grafo(session: ClientSession, testo_chunk: str, chunk_id: str, nome_materia: str):
    """Usa l'LLM per estrarre entità, relazioni e CLAIMS dal testo."""
    prompt = f"Estrai entità, relazioni e affermazioni chiave (claims) da questo testo. Non creare nodi Materia. Testo:\n{testo_chunk}"
    risultato_grafo = estrattore.invoke(prompt)

    # Salva i Nodi
    for nodo in risultato_grafo.nodi:
        await session.call_tool("crea_nodo", arguments={
            "macro_categoria": nodo.macro_categoria,
            "nome_entita": nodo.nome,
            "micro_categoria": nodo.micro_categoria,
            "descrizione_breve": nodo.descrizione_breve
        })
        
        # Collega il concetto al Chunk fisico (Memory Lineage)
        await session.call_tool("crea_relazione", arguments={
            "entita_origine": nodo.nome,
            "tipo_relazione": "MENZIONATO_IN",
            "entita_destinazione": chunk_id,
            "dettaglio": "Trovato nel chunk"
        })
        
        # Collega il concetto alla Materia
        await session.call_tool("crea_relazione", arguments={
            "entita_origine": nodo.nome,
            "tipo_relazione": "APPARTIENE_A",
            "entita_destinazione": nome_materia,
            "dettaglio": "Argomento correlato"
        })

    # Salva le Relazioni tra concetti
    for rel in risultato_grafo.relazioni:
        await session.call_tool("crea_relazione", arguments={
            "entita_origine": rel.origine,
            "tipo_relazione": rel.tipo_relazione,
            "entita_destinazione": rel.destinazione,
            "dettaglio": rel.dettaglio
        })

    #Salva i claims (affermazioni chiave) collegati al chunk
    for claim in risultato_grafo.claims:
        await session.call_tool("crea_claim", arguments={
            "origine_id": chunk_id, # Il chunk del PDF "sostiene" questa affermazione
            "testo_claim": claim.affermazione,
            "concetto_riferimento": claim.concetto_riferimento
        })

async def processa_singolo_documento(session: ClientSession, filepath: str, nome_file: str, nome_materia: str):
    """Orchestra la divisione in chunk, l'embedding e l'estrazione per un singolo PDF."""
    try:
        docs = PyPDFLoader(filepath).load()
        chunks = text_splitter.split_documents(docs)
        
        for i, chunk in enumerate(chunks):
            testo_chunk = chunk.page_content
            chunk_id = f"{nome_file}_chunk_{i}"

            print(f"    -> Elaborazione chunk {i+1}/{len(chunks)}...")
            
            # --- NOVITÀ 1: Filtro anti-spazzatura ---
            # Se il chunk è più corto di 40 caratteri (es. solo numero pagina o titolo), saltalo.
            if len(testo_chunk.strip()) < 40:
                print("      [Skip] Chunk troppo corto, ignorato per risparmiare token.")
                continue

            # --- NOVITÀ 2: Gestione dell'errore per SINGOLO CHUNK ---
            try:
                # 1. Pipeline Vettoriale
                await salva_chunk_e_vettore(session, chunk_id, testo_chunk, nome_file, nome_materia)
                
                # 2. Pipeline a Grafi
                await estrai_e_salva_grafo(session, testo_chunk, chunk_id, nome_materia)
                
            except Exception as e_chunk:
                # Se OpenAI fa i capricci su questo specifico chunk, lo saltiamo e proseguiamo!
                print(f"      ⚠️ Avviso LLM sul chunk {i+1} (ignorato): {str(e_chunk)}")
                
    except Exception as e:
        print(f"  ❌ Errore critico impossibile leggere il file {nome_file}: {str(e)}")

# ==========================================
# 3. FUNZIONE PRINCIPALE (ORCHESTRATORE)
# ==========================================
async def esegui_pipeline(cartella_base="upload"):
    """Punto di ingresso principale. Gestisce la connessione MCP, il controllo hash e naviga le cartelle."""
    print("🚀 Avvio Pipeline di Ingestione GraphRAG...\n")
    
    server_params = StdioServerParameters(command="python", args=[PERCORSO_SERVER_MCP])
    
    if not os.path.exists(cartella_base):
        print(f"❌ Errore: Cartella '{cartella_base}' non trovata.")
        return

    # Inizializza la connessione MCP
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("🔌 Connessione al Server MCP stabilita.")
            await session.initialize()

            # 1. Setup iniziale del database
            await inizializza_ontologia(session)

            # 2. Esplorazione cartelle
            for nome_materia in os.listdir(cartella_base):
                if nome_materia not in MATERIE_CONSENTITE: 
                    continue
                
                percorso_materia = os.path.join(cartella_base, nome_materia)
                if not os.path.isdir(percorso_materia): 
                    continue
                
                print(f"\n📘 Analisi Materia: {nome_materia}")
                
                # 3. Lettura documenti
                for file in os.listdir(percorso_materia):
                    if not file.endswith('.pdf'): 
                        continue
                        
                    filepath = os.path.join(percorso_materia, file)
                    
                    # --- MODIFICA: Controllo ID univoco tramite Hash SHA-256 ---
                    hash_corrente = calcola_hash_pdf(filepath)
                    
                    # Chiediamo al server se ha già visto l'hash di questo file
                    risposta_check = await session.call_tool("verifica_documento", arguments={"hash_file": hash_corrente})
                    
                    if "ESISTE" in risposta_check.content[0].text:
                        print(f"  ⏭️ Skip: {file} (Già elaborato in precedenza)")
                        continue
                    
                    print(f"  📄 Lettura e Ingestione: {file}")
                    
                    # Creiamo subito il nodo Documento e colleghiamolo alla Materia di appartenenza
                    await session.call_tool("registra_documento", arguments={
                        "nome_file": file,
                        "hash_file": hash_corrente,
                        "nome_materia": nome_materia
                    })
                    # -----------------------------------------------------------
                    
                    # Delega il lavoro pesante alla funzione dedicata
                    await processa_singolo_documento(session, filepath, file, nome_materia)
                            
            print("\n✅ Ingestione completata. Connessione MCP chiusa.")

if __name__ == "__main__":
    asyncio.run(esegui_pipeline())