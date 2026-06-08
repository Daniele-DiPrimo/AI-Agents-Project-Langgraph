from pydantic import BaseModel, Field
from typing import Literal, List

# --- 2. Schema di classificazione ---
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

class SingleSourceEvaluationSchema(BaseModel):
    id_source: str = Field(
        description="Sito o URL della fonte, tool utilizzato."
    )
    rate: float = Field(
        description="Un valore da 0.0 a 1.0 che indica l'affidabilità dei link e l'utilità dell'informazione rispetto all'argomento richiesto."
    )
    reasoning: str = Field(
        description="Spiega BREVEMENTE perchè hai assegnato questo punteggio"
    )

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