param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 9000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

python scripts\sanare_mcp_server.py --transport http --host $HostName --port $Port
