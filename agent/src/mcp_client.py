import os
from mcp import ClientSession, StdioServerParameters, stdio_client

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Inizializza la connessione al server MCP locale, esegue un tool
    specifico e ne restituisce il risultato testuale.
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
                return risultato.content[0].text
            
            return ""