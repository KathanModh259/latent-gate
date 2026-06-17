"""
LatentGate MCP Server — Module entry point
==========================================
Allows running via: python -m latent_gate.mcp_server

The actual server code is in integrations/mcp_server/server.py
This module just imports and runs it.
"""

import sys
import os

# Add integrations folder to path so we can import
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "integrations", "mcp_server")
)

from server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
