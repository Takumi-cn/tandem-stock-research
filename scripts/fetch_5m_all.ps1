# Fetch the full 5-minute history for the battery universe.
# Long running on purpose: run it as a background job, not in the foreground.
param(
  [string]$Root = '',
  [string]$Py = '',
  [string]$Start = '2021-10-01',
  [int]$Shards = 5
)
if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
if (-not $Py) { $Py = Join-Path $Root '.venv\Scripts\python.exe' }
$ErrorActionPreference = 'Continue'
Set-Location $Root
$env:PYTHONIOENCODING = 'utf-8'
$logdir = Join-Path $Root 'reports\micro005_minimal\logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir 'orchestrator.log'
function W($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

# 1. wait for the baostock public node (it is intermittently refused)
$ok = $false
for ($i = 1; $i -le 100; $i++) {
  $out = & $Py -c "import baostock as bs,sys; lg=bs.login(); print('LOGIN',lg.error_code); sys.exit(0 if lg.error_code=='0' else 3)" 2>&1 | Out-String
  if ($out -match 'LOGIN 0') { $ok = $true; W "login OK on probe $i"; break }
  $flat = ($out -replace '\s+', ' ').Trim()
  if ($flat.Length -gt 160) { $flat = $flat.Substring(0, 160) }
  W ("probe {0} failed: {1}" -f $i, $flat)
  Start-Sleep -Seconds 30
}
if (-not $ok) { W 'ABORT: baostock unreachable after 100 probes'; exit 2 }

# 2. shards
$procs = @()
for ($s = 0; $s -lt $Shards; $s++) {
  $a = @('-m', 'src.data.fetch_5m', '--shard', "$s", '--nshards', "$Shards", '--start', $Start)
  $procs += Start-Process -FilePath $Py -ArgumentList $a -WorkingDirectory $Root -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logdir "shard$s.out") -RedirectStandardError (Join-Path $logdir "shard$s.err")
}
W "launched $($procs.Count) shards"
$procs | Wait-Process
W 'all shards finished'

# 3. audit
$aud = & $Py -m src.data.fetch_5m --audit 2>&1 | Out-String
W ('audit: ' + (($aud -replace '\s+', ' ').Trim()))
W 'ORCHESTRATOR_DONE'
