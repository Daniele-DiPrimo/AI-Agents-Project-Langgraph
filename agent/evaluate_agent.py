import asyncio
from dotenv import load_dotenv
from langsmith import aevaluate
from langsmith.schemas import Run, Example
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Importa il tuo nodo e lo schema di stato
from src.agents import writer_node
from src.state import BlogState

load_dotenv()

# ==========================================
# 1. SETUP DEL GIUDICE (LLM-as-a-Judge)
# ==========================================
# Usiamo il modello come giudice inflessibile
judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class GraderOutput(BaseModel):
    score: int = Field(description="1 se il test è superato, 0 se fallito.")
    reason: str = Field(description="Spiegazione concisa del perché di questo voto.")

# ==========================================
# 2. DEFINIZIONE DEGLI EVALUATORS
# ==========================================

def grounding_evaluator(run: Run, example: Example) -> dict:
    """Valuta se l'articolo inventa informazioni non presenti nel materiale di ricerca."""
    
    # Estraiamo i dati di input (dal dataset) e l'output generato dal nodo
    research_material = example.inputs.get("research_material", "")
    final_article = run.outputs.get("final_article", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un valutatore esperto. Devi verificare il GROUNDING (fedeltà ai dati). "
                   "Confronta il MATERIALE DI RICERCA con l'ARTICOLO GENERATO. "
                   "L'articolo contiene informazioni tecniche, formule o tesi NON presenti nel materiale fornito? "
                   "Rispondi con score=1 se l'articolo è totalmente fedele e non allucina. "
                   "Rispondi con score=0 se ci sono allucinazioni o invenzioni."),
        ("human", f"MATERIALE DI RICERCA:\n{research_material}\n\nARTICOLO GENERATO:\n{final_article}")
    ])

    grader = judge_llm.with_structured_output(GraderOutput)
    chain = prompt | grader
    result = chain.invoke({})

    return {"key": "grounding_fidelity", "score": result.score, "comment": result.reason}


def citation_evaluator(run: Run, example: Example) -> dict:
    """Valuta se il writer ha inserito correttamente le citazioni in formato [Fonte]."""
    
    final_article = run.outputs.get("final_article", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un valutatore esperto. Verifica se il testo contiene citazioni espicite "
                   "nel formato [NomeFonte.pdf] o [http...]. Se ci sono affermazioni tecniche, "
                   "devono avere una citazione. Rispondi score=1 se le citazioni sono presenti "
                   "e formattate bene, score=0 se mancano o sono errate."),
        ("human", f"ARTICOLO GENERATO:\n{final_article}")
    ])

    grader = judge_llm.with_structured_output(GraderOutput)
    chain = prompt | grader
    result = chain.invoke({})

    return {"key": "citations_format", "score": result.score, "comment": result.reason}


# ==========================================
# 3. WRAPPER DEL TARGET (Il tuo Writer)
# ==========================================
async def target_writer_function(inputs: dict) -> dict:
    """Questa funzione simula il BlogState e lancia solo il tuo writer_node."""
    
    mock_state = BlogState(
        intent=inputs.get("intent", "ArticoloTeorico"),
        subject=inputs.get("subject", "Materia Sconosciuta"),
        specific_topic=inputs.get("specific_topic", "Argomento Sconosciuto"),
        prompt_to_reasoner="Scrivi in base al materiale",
        research_material=inputs.get("research_material", ""),
        # Mettiamo gli altri campi vuoti o di default per evitare errori
        messages=[],
        original_prompt="",
        suggestions=[],
        plan_justification="",
        current_suggestion_index=0,
        final_article="",
        graph_results={}
    )
    
    # Eseguiamo il tuo nodo asincrono
    result = await writer_node(mock_state)
    return {"final_article": result["final_article"]}

# ==========================================
# 4. ESECUZIONE DELLA VALUTAZIONE
# ==========================================
async def main():
    print("🚀 Avvio della valutazione su LangSmith...")
    
    dataset_name = "UniAgent_Evaluation_Dataset" # Assicurati che corrisponda al nome su LangSmith
    
    # Lancia l'evaluator. LangSmith prenderà ogni riga del dataset, la passerà a target_writer_function,
    # e poi passerà i risultati ai due evaluator (grounding e citation).
    experiment_results = await aevaluate(
        target_writer_function,
        data=dataset_name,
        evaluators=[grounding_evaluator, citation_evaluator],
        experiment_prefix="Writer_Quality_Test",
        max_concurrency=1
    )
    
    print("\n✅ Valutazione completata! Vai sulla dashboard di LangSmith per vedere i risultati.")

if __name__ == "__main__":
    asyncio.run(main())