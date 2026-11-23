$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$toolsPath = Join-Path (Get-Location) 'artifacts/tools'
$pythonPath = Join-Path $toolsPath 'reconstruction-python/Scripts/python.exe'
New-Item -ItemType Directory -Force -Path $toolsPath | Out-Null
if (!(Test-Path -LiteralPath $pythonPath)) {
    python -m venv (Join-Path $toolsPath 'reconstruction-python')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11–3.13 is required.' }
}
& $pythonPath -m pip install -r scripts/reconstruction-requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install reconstruction dependencies.' }
$brushPath = Join-Path $toolsPath 'brush/brush_app.exe'
if (!(Test-Path -LiteralPath $brushPath)) {
    $archive = Join-Path $toolsPath 'brush.zip'
    Invoke-WebRequest 'https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip' -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6') {
        throw 'Brush download checksum mismatch. Do not execute it.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $toolsPath 'brush') -Force
    if (!(Test-Path -LiteralPath $brushPath)) {
        $executable = Get-ChildItem (Join-Path $toolsPath 'brush') -Filter 'brush_app.exe' -Recurse | Select-Object -First 1
        if (!$executable) { throw 'Brush executable missing from verified archive.' }
        # Releases may contain a top-level package directory; keep the whole
        # directory in place and expose the executable at the documented path.
        Copy-Item -LiteralPath $executable.FullName -Destination $brushPath
    }
}
& $brushPath --version
if ($LASTEXITCODE -ne 0) { throw 'Brush could not start.' }
Write-Host 'Local tools ready. Source video, frames and models stay in ignored artifacts/.'
