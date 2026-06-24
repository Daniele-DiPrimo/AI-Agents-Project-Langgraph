import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
try: 
    driver_neo4j = GraphDatabase.driver(uri= os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
except Exception as e: 
    raise RuntimeError(f"Impossibile connettersi a neo4j. Error: {e}")


def neo4j_search(embedded_title, top_k) -> list[str]:

    query = """
    // 1. Punto di ingresso: Ricerca vettoriale sull'indice degli articoli
    CALL db.index.vector.queryNodes('article_vector_index', $top_k, $embedded_title)
    YIELD node AS a, score
    
    // 2. Filtro di sicurezza (Barriera anti-allucinazione)
    WHERE score >= $threshold
    
    // 3. Estrazione dei Concetti Spiegati
    OPTIONAL MATCH (a)-[:SPIEGA]->(c:Concetto_Teorico)
    
    // 4. Estrazione delle Affermazioni (Claims) relative a quei concetti
    OPTIONAL MATCH (a)-[:SOSTIENE]->(claim:Affermazione)-[:RIGUARDA]->(c)
    
    // 5. Estrazione dei Concetti Correlati (vicini nel grafo)
    OPTIONAL MATCH (c)-[]-(c_correlato:Concetto_Teorico)
    WHERE c_correlato <> c AND c_correlato IS NOT NULL
    
    // 6. Aggregazione dei risultati
    RETURN 
        a.nome AS article,
        score AS similarita,
        collect(DISTINCT c.nome) AS theorical_concepts,
        collect(DISTINCT claim.nome) AS key_claims,
        collect(DISTINCT c_correlato.nome) AS related_concepts
    ORDER BY score DESC
    """

    result = []

    try: 
        records, _, _ = driver_neo4j.execute_query(
            query, 
            parameters_={
                "embedded_title" : embedded_title,
                "top_k" : top_k,
                "threshold" : 0.80
            }
        )

        if not records:
            result = []
        else:
            for r in records:
                record = r.data()
                neo4j_result = f"Trovato articolo: {record['article']}\n"
                neo4j_result += f"Concetti correlati: {record['theorical_concepts']}"
                neo4j_result += f"Affermazioni Chiave: {record['key_claims']}\n"
                neo4j_result += f"Concetti Correlati: {record['related_concepts']}\n"
                neo4j_result += "--------------------------------"

                result.append(neo4j_result)
    
    except Exception as e: 
        print(f"Errore nella query Neo4j. {e}")
        result = []
    
    print(f"\n\n INFORMAZIONI DAL KNOWLEDGE GRAPH : {result}\n\n")
    return result
