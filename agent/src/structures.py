from pydantic import BaseModel, Field
from typing import Literal, List

class ClassificationSchema(BaseModel):
    intent: Literal["ArticoloTeorico", "TechNews", "Eserciziario"] = Field(
        description="L'intento dell'utente"
    )
    subject: str = Field(
        description="Il dominio generale o la materia dell'argomento "
    )
    specific_topic: str = Field(
        description="Il focus specifico e dettagliato dell'articolo"
    )
    prompt_to_reasoner: str = Field(
        description="Le istruzioni dettagliate e ripulite da passare al nodo reasoner"
    )

class SingleJudgment(BaseModel):
    source: str = Field(..., description="Il riferimenti alla fonte valutata")
    source_reliability: float
    source_relevance: float
    reasoning: str

class SourceEvaluationSchema(BaseModel):
    judgments: List[SingleJudgment]

class CompletenessEvaluationSchema(BaseModel): 
    is_complete: bool = Field(
        description="MANDATORIO: BOOLEANO (true/false). Restituisci true SOLO SE hai abbastanza paragrafi e concetti tecnici per scrivere un articolo esaustivo. Altrimenti false."
    )
    missing_info: str = Field(
        description="Se is_complete è False, elenca ESATTAMENTE cosa il planner deve cercare al prossimo giro. Se True, lascia vuoto."
    )

class SearchSchema(BaseModel):
    giustificazione: str = Field(
        description="OBBLIGATORIO: Spiega nel dettaglio il tuo ragionamento logico e PERCHÉ stai facendo questa ricerca."
    )
    query: str = Field(
        description="La query di ricerca in inglese."
    )

class RelazioneArticolo(BaseModel):
    origine: str = Field(description="Nome del concetto di origine")
    tipo_relazione: str = Field(description="SOLO TRA: [APPARTIENE_A, SI_BASA_SU, È_UN_TIPO_DI, COMPOSTO_DA, RISOLVE_USA, SPIEGA, SOSTIENE, RIGUARDA]")
    destinazione: str = Field(description="Nome del concetto di destinazione")
    dettaglio: str = Field(description="Contesto specifico in poche parole")

class ClaimArticolo(BaseModel):
    affermazione: str = Field(description="Una frase completa che esprime una tesi, una regola o un fatto chiave (max 15 parole).")
    concetto_riferimento: str = Field(description="Il nome esatto del concetto teorico a cui si riferisce.")

class EstrazioneMetadatiArticolo(BaseModel):
    concetti_trovati: List[str] = Field(description="Lista dei concetti teorici spiegati nell'articolo.")
    relazioni_concetti: List[RelazioneArticolo] = Field(description="Relazioni logiche tra i concetti trovati in questo articolo.")
    fonti: List[str] = Field(description="Lista di tutte le fonti utilizzate per scrivere l'articolo")
    claims_estratti: List[ClaimArticolo] = Field(description="Le affermazioni chiave o conclusioni fatte nell'articolo.") 

class QueryExpansionSchema(BaseModel):

    is_context_relevant: bool = Field(
        description="True se il contesto storico di Neo4j è effettivamente correlato al nuovo argomento richiesto. False se è un falso positivo o parla di tutt'altro."
    )
    reasoning: str = Field(
        description="Spiega in una riga perché il contesto è rilevante o meno."
    )
    queries: list[str] = Field(
        description="Lista di 3 query di ricerca. Se is_context_relevant è False, ignora il contesto storico e formula query partendo da zero sull'argomento richiesto."
    )

class ChromaQuerySchema(BaseModel):
    queries: list[str] = Field(
        description="Esattamente 3 query di ricerca ottimizzate per Chroma DB"
    )