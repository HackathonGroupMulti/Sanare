from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sanare-mcp.out.log"
ERR = ROOT / "sanare-mcp.err.log"


def main() -> None:
    out = OUT.open("ab")
    err = ERR.open("ab")
    subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "sanare_mcp_server.py"),
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ],
        cwd=ROOT,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    print("Sanare MCP spawn requested on http://127.0.0.1:9000/mcp/")


if __name__ == "__main__":
    main()
