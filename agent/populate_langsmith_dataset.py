from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

# Inizializza il client LangSmith (assicurati che LANGCHAIN_API_KEY sia nel .env)
client = Client()
dataset_name = "UniAgent_Evaluation_Dataset"

# Crea il dataset se non esiste
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name, description="Dataset per test di Grounding e Citazioni (Testi >30 righe)")
else:
    dataset = client.read_dataset(dataset_name=dataset_name)

print(f"Dataset '{dataset_name}' trovato/creato. Inizio l'inserimento degli esempi...")

# =========================================================
# DEFINIZIONE DEI 5 ESEMPI (Testi lunghi >30 righe)
# =========================================================
examples = [
    {
        "intent": "ArticoloTeorico",
        "subject": "Sistemi Operativi",
        "specific_topic": "La gestione della memoria: Paging e TLB",
        "research_material": """# Contesto Estratto dal Knowledge Graph [K-RAG_OS_Concepts]
            Il paging è uno schema di gestione della memoria logica che permette allo spazio degli indirizzi fisici di un processo di essere non contiguo. 
            Questo risolve i grandi problemi legati alla frammentazione esterna e alla necessità di compattazione della memoria.
            La memoria fisica viene divisa in blocchi di dimensione fissa chiamati 'frame'.
            La memoria logica viene divisa in blocchi della stessa dimensione chiamati 'pagine' (pages).
            Le dimensioni delle pagine sono solitamente potenze di 2, tipicamente tra i 4 KB e gli 8 KB nelle architetture moderne.

            # Approfondimento da Fonte Esterna [Silberschatz_Cap8.pdf]
            Quando la CPU genera un indirizzo logico, questo è diviso in due parti: il numero di pagina (p) e l'offset di pagina (d).
            Il numero di pagina è usato come indice in una tabella delle pagine (Page Table).
            La tabella delle pagine contiene l'indirizzo di base di ogni pagina nella memoria fisica.
            Questo indirizzo di base è combinato con l'offset per definire l'indirizzo fisico completo.
            L'uso della Page Table introduce un problema di prestazioni: ogni accesso ai dati richiede due accessi in memoria (uno per la tabella, uno per il dato).
            Per risolvere questo problema, le architetture moderne utilizzano una cache hardware speciale e velocissima chiamata TLB (Translation Look-aside Buffer).
            La TLB è una memoria associativa. Quando un indirizzo logico viene generato, il numero di pagina viene presentato alla TLB.
            Se il numero di pagina viene trovato (TLB hit), il frame corrispondente è immediatamente disponibile.
            Se non viene trovato (TLB miss), si verifica l'accesso alla Page Table in memoria centrale, e successivamente la TLB viene aggiornata.

            # Dettagli prestazionali [Semantic_Scholar_Paper_TLB_2022]
            I moderni processori Intel Core supportano più livelli di TLB (L1 e L2 TLB).
            Il tasso di TLB hit (hit ratio) supera tipicamente il 99% per applicazioni ben ottimizzate.
            Un TLB miss può causare una penalità dai 10 ai 100 cicli di clock, a seconda se la page table entry (PTE) si trova nella cache L1, L2 o nella RAM.
            Inoltre, alcuni sistemi supportano 'Huge Pages' (es. 2 MB o 1 GB) per ridurre il numero di entry necessarie nella TLB e abbassare il miss rate per applicazioni data-intensive come i database relazionali."""
    },
    {
        "intent": "Eserciziario",
        "subject": "Analisi Matematica I",
        "specific_topic": "Studio del Dominio e Asintoti di Funzioni Fratte",
        "research_material": """# Fondamenti Teorici [Dispense_Analisi1_PoliMi.pdf]
            Lo studio di una funzione reale di variabile reale inizia sempre con la determinazione del suo Dominio (o Insieme di Definizione).
            Per le funzioni razionali fratte della forma f(x) = N(x) / D(x), il dominio si ottiene imponendo il denominatore D(x) diverso da zero.
            Ad esempio, se D(x) è un polinomio di secondo grado, occorre risolvere l'equazione D(x) = 0 per trovare i punti di discontinuità.
            I punti esclusi dal dominio sono i candidati principali per la ricerca di asintoti verticali.

            # Ricerca degli Asintoti [Marcellini_Sbordone_Vol1.pdf]
            Un asintoto verticale in x = x0 esiste se il limite di f(x) per x che tende a x0 è infinito (+ o - infinito).
            Per gli asintoti orizzontali, si calcola il limite di f(x) per x che tende a infinito.
            Se il grado del numeratore N(x) è uguale al grado del denominatore D(x), l'asintoto orizzontale esiste e la sua equazione è y = k, dove k è il rapporto tra i coefficienti di grado massimo.
            Se il grado del numeratore è superiore di 1 rispetto al denominatore, può esistere un asintoto obliquo di equazione y = mx + q.
            Il coefficiente angolare m si trova calcolando il limite per x che tende a infinito di f(x)/x.
            L'intercetta q si trova calcolando il limite per x che tende a infinito di [f(x) - mx].

            # Esempio Svolto 1 [Eserciziario_Analisi_Web.html]
            Data la funzione f(x) = (x^2 - 4) / (x - 1).
            Il dominio è tutto l'asse reale escluso x = 1. In intervalli: (-infinito, 1) unione (1, +infinito).
            Calcolando il limite per x -> 1, si ottiene N(1) = -3 e D(1) = 0, quindi il limite è infinito. La retta x = 1 è asintoto verticale.
            Poiché il numeratore ha grado 2 e il denominatore grado 1, non ci sono asintoti orizzontali.
            Calcoliamo m per l'asintoto obliquo: lim_{x->inf} f(x)/x = lim_{x->inf} (x^2 - 4) / (x^2 - x) = 1. Quindi m = 1.
            Calcoliamo q: lim_{x->inf} [f(x) - x] = lim_{x->inf} [(x^2 - 4 - x(x - 1)) / (x - 1)] = lim_{x->inf} (x - 4) / (x - 1) = 1. Quindi q = 1.
            L'equazione dell'asintoto obliquo è y = x + 1."""
    },
    {
        "intent": "TechNews",
        "subject": "Machine Learning",
        "specific_topic": "L'evoluzione dell'architettura Transformer e le context window estese",
        "research_material": """# Ultime Notizie AI [Tavily_AI_News_Weekly]
            Negli ultimi mesi, la gara tra le grandi aziende tecnologiche (Google, OpenAI, Anthropic) si è spostata dalla pura grandezza dei modelli (numero di parametri) alla dimensione della 'context window' (finestra di contesto).
            La context window rappresenta il numero massimo di token (parole o frammenti di parole) che un modello linguistico può elaborare contemporaneamente in una singola richiesta.
            Fino al 2022, lo standard industriale era di 4.096 token. Oggi stiamo assistendo a modelli capaci di gestire da 128.000 fino a 2 milioni di token.

            # Dettagli Tecnici sui Transformer [Attention_Is_All_You_Need_V2.pdf]
            L'architettura Transformer originale ha una limitazione fondamentale: il meccanismo di 'self-attention' scala quadraticamente rispetto alla lunghezza della sequenza.
            Questo significa che raddoppiare i token in ingresso quadruplica il tempo di calcolo e la memoria RAM necessaria.
            Per aggirare questo ostacolo matematico, i ricercatori hanno sviluppato nuove tecniche.
            Una delle più promettenti è la 'Ring Attention', che distribuisce il calcolo dell'attenzione su più GPU simultaneamente, permettendo sequenze teoricamente infinite.
            Un'altra tecnica vitale è il RoPE (Rotary Position Embedding), che migliora il modo in cui il modello comprende la posizione relativa delle parole nelle lunghe sequenze.

            # Casi d'uso e Benchmark [Semantic_Scholar_NeedleInAHaystack.pdf]
            La metrica principale per valutare le finestre di contesto estese è il test 'Needle In A Haystack' (Ago in un pagliaio).
            In questo test, un fatto minuscolo (l'ago) viene nascosto all'interno di un documento di 1 milione di parole (il pagliaio), e al modello viene chiesto di recuperarlo.
            Modelli recenti come Gemini 1.5 Pro hanno dimostrato un tasso di recupero del 99% fino a 1 milione di token.
            Questa capacità sta rivoluzionando l'uso della RAG (Retrieval-Augmented Generation).
            Invece di frammentare i documenti e cercare solo le parti rilevanti (chunking), le aziende stanno iniziando a caricare interi libri o decine di paper scientifici direttamente nel prompt del modello, eliminando gli errori di recupero tipici dei database vettoriali."""
    },
    {
        "intent": "ArticoloTeorico",
        "subject": "Database",
        "specific_topic": "La Teoria della Normalizzazione: Dalla 1NF alla BCNF",
        "research_material": """# Concetti Fondamentali [K-RAG_DB_Normalization]
            La normalizzazione è un processo sistematico utilizzato nella progettazione di database relazionali.
            Il suo scopo principale è eliminare la ridondanza dei dati e prevenire anomalie di inserimento, aggiornamento e cancellazione (anomalie DML).
            Il processo si basa sul concetto di dipendenza funzionale (X -> Y), ovvero: se conosco il valore dell'attributo X, posso determinare univocamente il valore dell'attributo Y.
            La normalizzazione avviene in fasi sequenziali, denominate Forme Normali (NF).

            # Prima e Seconda Forma Normale [Elmasri_Navathe_DBSystems.pdf]
            Una tabella è in Prima Forma Normale (1NF) se e solo se tutti i suoi attributi sono atomici. 
            Questo significa che non sono ammessi array, liste o attributi multivalore all'interno di una singola cella della tabella.
            La Seconda Forma Normale (2NF) richiede che la tabella sia in 1NF e che ogni attributo non chiave dipenda dall'INTERA chiave primaria, non solo da una parte di essa.
            La 2NF si applica principalmente quando si hanno chiavi primarie composte (formate da due o più attributi). Se un attributo dipende solo da mezza chiave, si parla di 'dipendenza parziale' e bisogna dividere la tabella.

            # Terza Forma Normale e BCNF [Database_Design_Codd_1970.pdf]
            Una tabella è in Terza Forma Normale (3NF) se è in 2NF e non presenta dipendenze transitive.
            Una dipendenza transitiva si ha quando un attributo non chiave dipende da un altro attributo non chiave (es. Impiegato -> Dipartimento -> Sede del Dipartimento).
            Per risolvere la 3NF, la regola pratica è: "Ogni attributo deve fornire informazioni sulla chiave, l'intera chiave e nient'altro che la chiave".
            Esiste una forma ancora più rigorosa, la Forma Normale di Boyce-Codd (BCNF).
            Una tabella è in BCNF se per ogni dipendenza funzionale non banale X -> Y, X è una superchiave.
            La BCNF è una versione potenziata della 3NF che gestisce i casi rari in cui una tabella ha più chiavi candidate sovrapposte. Nella pratica aziendale, fermarsi alla 3NF è quasi sempre sufficiente per le prestazioni."""
    },
    {
        "intent": "Eserciziario",
        "subject": "Fondamenti di Programmazione",
        "specific_topic": "Algoritmi di Ordinamento: Il Merge Sort in Python",
        "research_material": """# Basi Teoriche del Merge Sort [Algoritmi_Cormen_Intro.pdf]
            Il Merge Sort è un algoritmo di ordinamento basato sul paradigma 'Divide et Impera' (Divide and Conquer).
            È stato inventato da John von Neumann nel 1945.
            L'algoritmo funziona dividendo ripetutamente l'array non ordinato in due metà, fino a quando ogni sotto-array contiene un solo elemento.
            Un array di un solo elemento è, per definizione, già ordinato.
            Successivamente, la funzione 'merge' viene utilizzata per unire (fondere) i sotto-array in array più grandi e ordinati, fino a ricostruire l'array originale completo.

            # Analisi della Complessità [Appunti_Algoritmi_PoliTo.pdf]
            A differenza del Bubble Sort o dell'Insertion Sort che hanno un tempo di esecuzione quadratico nel caso pessimo (O(n^2)), il Merge Sort è estremamente efficiente.
            Il suo tempo di esecuzione è garantito essere O(n log n) in tutti i casi (migliore, medio e pessimo).
            Tuttavia, ha uno svantaggio spaziale: non ordina i dati 'in-place'.
            Richiede una memoria ausiliaria aggiuntiva proporzionale alla grandezza dell'array da ordinare, quindi la sua complessità spaziale è O(n).

            # Esercizio Pratico e Traccia Codice [Dispense_Python_Coding.html]
            Si richiede l'implementazione del Merge Sort in linguaggio Python.
            La struttura base richiede due funzioni: `merge_sort(arr)` e `merge(left, right)`.
            Passo 1: Nella funzione `merge_sort`, si trova il punto medio: `mid = len(arr) // 2`.
            Passo 2: Si creano le due metà: `left_half = arr[:mid]` e `right_half = arr[mid:]`.
            Passo 3: Si richiamano ricorsivamente: `left_half = merge_sort(left_half)`.
            Passo 4: Nella funzione `merge`, si usano due puntatori (i, j) per scorrere left e right.
            Si confrontano `left[i]` e `right[j]`. Il minore viene appeso alla lista risultato `merged`, e si incrementa il rispettivo puntatore.
            Infine, se rimangono elementi residui in una delle due liste, si appendono in blocco alla fine della lista risultato.
            Esempio di input per l'esercizio: arr = [38, 27, 43, 3, 9, 82, 10].
            Output atteso: [3, 9, 10, 27, 38, 43, 82]."""
    }
]

# Inserimento iterativo nel dataset
count = 0
for ex in examples:
    # Salviamo l'intero dizionario negli 'inputs' previsti dall'evaluate_agent
    client.create_example(
        inputs={
            "intent": ex["intent"],
            "subject": ex["subject"],
            "specific_topic": ex["specific_topic"],
            "research_material": ex["research_material"]
        },
        outputs={},  # Per l'LLM-as-a-judge non serve un ground truth esatto qui
        dataset_id=dataset.id
    )
    count += 1

print(f"✅ Inserimento completato con successo: {count} esempi aggiunti a '{dataset_name}'.")
print("Ora puoi lanciare il tuo script `evaluate_agent.py`!")