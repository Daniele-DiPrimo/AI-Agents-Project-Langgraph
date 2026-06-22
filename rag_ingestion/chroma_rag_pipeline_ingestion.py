import os
import time
from dotenv import load_dotenv
from pathlib import Path
import chromadb 
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm 
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from google import genai
import hashlib

OUTPUT_MD_DIR = "./output_md"
Path(OUTPUT_MD_DIR).mkdir(exist_ok=True)

# Trova la root del progetto partendo dal file corrente 
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

load_dotenv(dotenv_path=env_path)

print(f"CHROMA_HOST: {os.getenv('CHROMA_HOST')}")

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

google_client = genai.Client()

google_embedding = embedding_functions.GoogleGenaiEmbeddingFunction()

collection = chroma_client.get_or_create_collection(name="UniAgent_RAG", embedding_function=google_embedding)

vision_model = "gemini-2.5-flash-lite"

def save_markdown(md_text: str, filename: str):
    """Salva il contenuto Markdown su file per debug."""
    output_path = Path(OUTPUT_MD_DIR) / f"{filename}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"   💾 Markdown salvato in: {output_path}")

def get_file_hash(file_path: str) -> str:
    """Calcola l'hash SHA-256 leggendo i byte del file originale."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_file_already_ingested(file_hash: str) -> bool:
    """Interroga ChromaDB per vedere se esistono chunk associati a questo file."""
    try:
        # Cerca nel database almeno un documento che abbia questo hash nei metadati
        results = collection.get(
            where={"doc_hash": file_hash},
            limit=1,
            include=["metadatas"]
        )
        return len(results["ids"]) > 0
    except Exception as e:
        print(f"Errore durante il controllo nel DB: {e}")
        return False

def get_files_to_process():
    """
    Scansiona la cartella upload e restituisce una lista di tuple: (percorso_file, subject)
    """
    files_to_process = []
    base_path = Path("./upload")
    
    for subject_dir in base_path.iterdir():
        if subject_dir.is_dir():
            subject = subject_dir.name
            # Itera sui file dentro la cartella della subject
            for file_path in subject_dir.iterdir():
                if file_path.is_file():
                    files_to_process.append((str(file_path), subject))
                    
    return files_to_process

def convert_file_to_markdown(file_path: str) -> str:
    """
    Invia il file a Gemini e richiede la conversione in Markdown strutturato.
    """
    sys_prompt = """
    Sei un estrattore e convertitore avanzato di documenti universitari scientifici.
    Trascrivi fedelmente tutto il testo strutturandolo in un Markdown pulito e gerarchico. Questo testo verrà successivamente frammentato in blocchi logici, quindi la formattazione è vitale.

    REGOLE FONDAMENTALI:
    1. GERARCHIA: Usa rigorosamente e coerentemente i tag di intestazione (# Capitolo, ## Paragrafo, ### Sottoparagrafo). Assicurati di non saltare i livelli logici.
    Assicurati che ogni sezione non superi le 300-500 parole. Se una sezione è troppo lunga, crea dei sottoparagrafi usando ## o ###. Il tuo obiettivo è produrre un file che abbia almeno 5-10 titoli (##) per permettere una corretta frammentazione.
    2. MATEMATICA: Converti TUTTE le formule matematiche, equazioni, limiti, integrali e simboli speciali in codice LaTeX puro.
    3. SINTASSI LATEX: Usa il dollaro singolo ($) per le formule in linea e il doppio dollaro ($$) per le equazioni isolate/blocchi.
    4. GRAFICI: Se incontri grafici o diagrammi a blocchi, descrivili brevemente a parole racchiudendo il testo tra parentesi quadre (es. [Grafico: andamento della funzione...]).
    5. TABELLE: Se ci sono tabelle reali nel testo, convertile nel formato tabella standard di Markdown.
    6. RUMORE DI FONDO: IGNORA tassativamente le intestazioni e i piè di pagina ripetitivi (es. numeri di pagina, "Versione Provvisoria", nome del professore).

    RISPONDI SOLO ED ESCLUSIVAMENTE CON IL TESTO IN MARKDOWN PURO. NESSUN SALUTO, NESSUNA INTRODUZIONE, NESSUN RAGIONAMENTO LOGICO.
    """

    # 1. upload del file
    file_up = google_client.files.upload(file=file_path)

    while file_up.state.name == "PROCESSING":
        print("   Attendendo l'elaborazione del file lato server...")
        time.sleep(2)
        # Aggiorna lo stato interrogando nuovamente il file
        file_up = google_client.files.get(name=file_up.name)

    MAX_ATTEMPTS = 3
    md_text = ""

    for attempt in range(MAX_ATTEMPTS):
        try:
            conversion_result = google_client.models.generate_content(
                model=vision_model,
                contents=[file_up, sys_prompt]
            )
            md_text = conversion_result.text
            stop_reason = conversion_result.candidates[0].finish_reason
            print(f"\nESITO TESTO: {stop_reason.name}. SE MAX_TOKEN, PDF TROPPO LUNGO!\n")
            break 
            
        except Exception as e_doc:
            error = str(e_doc)
            
            if "429" in error or "RESOURCE_EXHAUSTED" in error:
                print(f"      ⏳ Quota OCR raggiunta (429). Pausa di 30 secondi e riprovo...")
                time.sleep(30)
            elif "503" in error:
                print(f"      ⚠️ Server Google intasati (503). Pausa di 10 secondi e riprovo...")
                time.sleep(10)
            else:
                google_client.files.delete(name=file_up.name)
                raise Exception(f"Impossibile connettersi a Gemini dopo {MAX_ATTEMPTS} tentativi: {error}")
    
    google_client.files.delete(name=file_up.name)

    return md_text

def chunk_markdown_dynamic(md_text: str, subject: str, filename: str, file_hash: str, max_chars: int = 1500) -> list[dict]:
    """
    Logica a Cascata (Inception): 
    L1 (#) -> Se troppo lungo -> L2 (##) -> Se troppo lungo -> L3 (###) 
    -> Se ANCORA troppo lungo -> Taglio netto a caratteri.
    """
    chunks_data = []
    
    ultimate_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    def save_chunk(text, metadata):
        final_metadata = metadata.copy()
        final_metadata["subject"] = subject
        final_metadata["source"] = filename
        final_metadata["doc_hash"] = file_hash # Iniettiamo l'hash del documento qui
        chunks_data.append({
            "text": text,
            "metadata": final_metadata
        })

    # --- LIVELLO 1: Capitoli (#) ---
    l1_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
    chunks_l1 = l1_splitter.split_text(md_text)
    
    for c1 in chunks_l1:
        if len(c1.page_content) <= max_chars:
            save_chunk(c1.page_content, c1.metadata)
        else:
            
            # --- LIVELLO 2: Paragrafi (##) ---
            l2_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Header 2")])
            chunks_l2 = l2_splitter.split_text(c1.page_content)
            
            for c2 in chunks_l2:
                meta2 = c2.metadata.copy()
                meta2.update(c1.metadata)
                
                if len(c2.page_content) <= max_chars:
                    save_chunk(c2.page_content, meta2)
                else:
                    
                    # --- LIVELLO 3: Sottoparagrafi (###) ---
                    l3_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "Header 3")])
                    chunks_l3 = l3_splitter.split_text(c2.page_content)
                    
                    for c3 in chunks_l3:
                        meta3 = c3.metadata.copy()
                        meta3.update(meta2)
                        
                        if len(c3.page_content) <= max_chars:
                            save_chunk(c3.page_content, meta3)
                        else:
                            
                            # --- FALLBACK FINALE: Taglio a caratteri ---
                            print(f"   ⚠️ Sottoparagrafo critico rilevato (> {max_chars} char). Applico fallback di emergenza.")
                            emergency_chunks = ultimate_splitter.split_text(c3.page_content)
                            
                            for e_chunk in emergency_chunks:
                                save_chunk(e_chunk, meta3)
                                
    print(f"\n✅ Elaborazione completata: generati {len(chunks_data)} chunk gerarchici.")
    return chunks_data

def ingest_into_chroma(chunks_data: list[dict]):
    """
    Riceve i chunk elaborati e li salva in ChromaDB tramite upsert.
    L'ID univoco del chunk deriva dall'hash del documento padre + indice.
    """
    if not chunks_data:
        return

    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks_data):
        chunk_text = chunk["text"]
        chunk_metadata = chunk["metadata"]
        
        # Generiamo l'ID usando l'hash del file originale + l'indice posizionale
        doc_hash = chunk_metadata["doc_hash"]
        chunk_id = f"{doc_hash}_chunk_{i}"
        
        documents.append(chunk_text)
        metadatas.append(chunk_metadata)
        ids.append(chunk_id)
        
    try:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"   ✅ Inseriti/Aggiornati {len(ids)} chunk nel database.")
    except Exception as e:
        print(f"   ❌ Errore durante l'ingestion in ChromaDB: {e}") 

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_pipeline():
    print("Inizio scansione cartelle...")
    files = get_files_to_process()
    print(f"Trovati {len(files)} file da processare.")
    
    for file_path, subject in tqdm(files, desc="Elaborazione file"):
        try:
            filename = Path(file_path).name
            
            # --- NOVITÀ: Hashing preventivo ---
            print(f"\nControllo file: {filename}")
            file_hash = get_file_hash(file_path)
            
            if is_file_already_ingested(file_hash):
                print(f"   ⏭️ File già presente nel Database vettoriale (Hash match). Skipping...")
                continue # Salta all'iterazione successiva del ciclo for
            
            # 1. Conversione
            md_text = convert_file_to_markdown(file_path)
            
            # 2. Chunking e Metadati
            if md_text: 
                save_markdown(md_text, filename)
                # Passiamo il file_hash alla funzione di chunking
                chunks = chunk_markdown_dynamic(md_text, subject, filename, file_hash)
                
                # 3. Ingestion
                if chunks:
                    ingest_into_chroma(chunks)
                    
        except Exception as e:
            print(f"\nErrore durante l'elaborazione di {file_path}: {e}")

if __name__ == "__main__":
    Path("./upload").mkdir(exist_ok=True)
    run_pipeline()