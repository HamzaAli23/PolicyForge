$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path build | Out-Null
$core = @("src/gridworld.cpp", "src/algorithms.cpp", "src/report.cpp")

& g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread -Iinclude @core "tests/test_main.cpp" -o "build/policyforge-tests.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& ".\build\policyforge-tests.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
