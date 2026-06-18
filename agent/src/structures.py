from pydantic import BaseModel, Field
from typing import Literal, List

class ClassificationSchema(BaseModel):
    intent: Literal["News", "Teoria", "Esercizio"] = Field(
        description="L'intento dell'utente"
    )
    macro_domain: str = Field(
        description="Il dominio generale o la materia dell'argomento "
    )
    specific_topic: str = Field(
        description="Il focus specifico e dettagliato dell'articolo"
    )
    prompt_to_reasoner: str = Field(
        description="Le istruzioni dettagliate e ripulite da passare al nodo reasoner"
    )

class SingleSourceEvaluationSchema(BaseModel):
    source_reliability: float = Field(
        description="Da 0.0 a 1.0. Valuta l'autorevolezza e l'affidabilità della fonte (es. siti ufficiali, paper = alto; forum sconosciuti = basso)."
    )
    information_relevance: float = Field(
        description="Da 0.0 a 1.0. Valuta quanto l'informazione è pertinente, utile e centrata rispetto all'argomento cercato."
    )
    reasoning: str = Field(
        description="Spiega BREVEMENTE perchè hai assegnato questo punteggio"
    )
    index_source: int = Field(
        description="L'ID numerico esatto della fonte valutata (es. 0, 1, 2)")

class FullSourcesEvaluationSchema(BaseModel):
    judgments: List[SingleSourceEvaluationSchema] = Field(
        description="La lista delle valutaioni per ogni singola informazione resistuita"
    )

    need_new_search: bool = Field(
        description="True se tutte le fonti sono state scartate e serce un approccio completamente diverso. False se almeno una fonte utile è stata trovata"
    )


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
    fonti_documentali: List[str] = Field(description="Lista esatta dei nomi dei file PDF citati nel testo (es. 'Algoritmi_Cap3.pdf').")
    link_esterni: List[str] = Field(description="Lista di tutti gli URL o link web menzionati nel testo. Lista vuota se non ce ne sono.")
    claims_estratti: List[ClaimArticolo] = Field(description="Le affermazioni chiave o conclusioni fatte nell'articolo.") # <-- NUOVO