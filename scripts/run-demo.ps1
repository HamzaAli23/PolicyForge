$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path build | Out-Null
$sources = @("src/gridworld.cpp", "src/algorithms.cpp", "src/report.cpp", "src/main.cpp")

& g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread -Iinclude @sources -o "build/policyforge.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& ".\build\policyforge.exe" --environment "examples/campus-navigation.env" --output "runs/demo" --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Start-Process "runs\demo\report.html"
