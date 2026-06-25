"""import os
import json
from typing import Any
from mcp import ClientSession, StdioServerParameters, stdio_client

async def call_mcp_tool(tool_name: str, arguments: dict) -> Any:
"""
    #Inizializza la connessione al server MCP locale, esegue un tool
    #specifico e ne restituisce il risultato (str, list, dict o altro).
"""
    percorso_mcp = os.getenv("MCP_SERVER_URI", "")
    
    if not percorso_mcp:
        raise ValueError("ERRORE: Variabile d'ambiente 'MCP_SERVER_URI' non trovata.")

    server_params = StdioServerParameters(command="python", args=[percorso_mcp])
    
    # Gestione contestuale di connessione e sessione
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Chiamata effettiva al tool
            risultato = await session.call_tool(tool_name, arguments=arguments)
            
            # Estrazione sicura del testo dalla risposta MCP
            if risultato.content and len(risultato.content) > 0:
                testo_raw = risultato.content[0].text
                
                # Tenta di decodificare stringhe JSON in dict o list
                try:
                    return json.loads(testo_raw)
                except json.JSONDecodeError:
                    # Se non è un JSON valido (è una stringa semplice), ritorna la stringa
                    return testo_raw
            
            return ""
"""

import os
import json
from typing import Any
from mcp import ClientSession
from mcp.client.sse import sse_client # <-- Cambiato import

async def call_mcp_tool(tool_name: str, arguments: dict) -> Any:
    # Ora passiamo un URL, non un percorso file!
    mcp_url = "http://localhost:8000/sse"
    
    print("🌐 [Neo4j] Connessione MCP via Rete (SSE)...")
    
    # Niente più StdioServerParameters, niente python executable
    async with sse_client(mcp_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            risultato = await session.call_tool(tool_name, arguments=arguments)
            
            if risultato.content and len(risultato.content) > 0:
                testo_raw = risultato.content[0].text
                try:
                    return json.loads(testo_raw)
                except json.JSONDecodeError:
                    return testo_raw
            return ""