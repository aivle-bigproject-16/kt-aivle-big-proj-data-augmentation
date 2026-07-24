<#
  run_pipeline.ps1
  quality-fail-augment 파이프라인의 단일 실행 진입점이다.

  ## 스테이지

      plan     결정론적 plan 생성(병렬 scan). generation_plan.csv 를 만든다.
      smoke    소규모(기본 100장/모달리티) 생성으로 worker_peak_rss_bytes 를 측정하고,
               측정값을 넣은 <config>.measured.json 을 만든다. 이 config 로 plan 을 다시
               만들면 이후 full 생성이 병렬 worker 로 돈다.
      generate 승인된 plan 으로 전체 생성. 완료 후 fail_visual_qa.csv 가 만들어지고
               "Visual QA approval pending" 으로 멈춘다(정상). 사람이 510장을 검토해
               reviewer 와 approved 를 채워야 한다. 이 단계는 사람 게이트를 대신 통과할
               수 없다.
      resume   사람이 QA CSV 를 채운 뒤 실행한다. case 별 승인율이 95% 이상이면 최종
               ZIP 과 완료 summary 를 공개한다.
      verify   생성된 데이터셋을 재검증한다.
      upload   생성 산출물을 rclone 으로 원격(gdrive 등)에 올린다.

  ## 참고 레포와 다른 점

  전처리 파이프라인의 approve 단계는 사람 이름만 기록하면 통과하지만, 이 레포의
  visual-QA 게이트는 사람이 실제로 510장을 보고 case 별 95% 승인을 채워야 한다.
  스크립트는 그 검토를 자동화하지 않는다. generate 가 멈추면 사람이 CSV 를 채운 뒤
  resume 를 부른다.

  ## 반드시 pwsh(PowerShell 7)로 실행한다

  Windows PowerShell 5.1은 BOM 없는 UTF-8 파일을 ANSI로 읽어 한글 경로를 깨뜨린다.

  ## 장시간 실행은 분리(-Detached)한다

  plan·smoke·generate 는 길게는 여러 시간이 걸린다. 에이전트 세션의 자식 프로세스는
  툴 호출이 끝나면 정리되므로, -Detached 로 스스로를 독립 프로세스로 다시 띄운다.

  ## 진행 상황

  단계마다 <Output>\pipeline.status 에 시작과 종료를 기록한다. 마지막 줄이
  finished ... exit=0 이면 정상 완료다. exit=10 은 사람 QA 게이트 대기를 뜻한다.

  ## 사용 예

      pwsh -File .\run_pipeline.ps1 -Stage plan     -RawRoot "..." -Config .\config.40k.json -Output "D:\qf_plan" -Detached
      pwsh -File .\run_pipeline.ps1 -Stage smoke    -RawRoot "..." -Config .\config.40k.json -Plan "D:\qf_plan\manifests\generation_plan.csv" -Output "D:\qf_smoke" -Detached
      pwsh -File .\run_pipeline.ps1 -Stage generate -RawRoot "..." -Config .\config.measured.json -Plan "D:\qf_plan2\manifests\generation_plan.csv" -Output "D:\qf_full" -Detached
      pwsh -File .\run_pipeline.ps1 -Stage resume   -RawRoot "..." -Config .\config.measured.json -Plan "D:\qf_plan2\manifests\generation_plan.csv" -Output "D:\qf_full" -Detached
      pwsh -File .\run_pipeline.ps1 -Stage verify   -Output "D:\qf_full"
      pwsh -File .\run_pipeline.ps1 -Stage upload   -Output "D:\qf_full" -Remote "gdrive:quality_fail_40k_v1.5" -Detached
#>
param(
    [ValidateSet("plan", "smoke", "generate", "resume", "verify", "upload")]
    [Parameter(Mandatory = $true)] [string]$Stage,
    [string]$RawRoot,
    [string]$Config,
    [string]$Plan,
    [string]$Output,
    [string]$Remote,
    [int]$LimitPerModality = 100,
    [switch]$TrustPlan,
    [switch]$KeepDisplay,
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

# 긴 단계는 호출 세션의 자식이 되면 안 된다. -Detached 로 독립 프로세스로 다시 띄운다.
# Start-Process -ArgumentList 는 공백을 포함한 배열 원소를 인용하지 않고 공백으로 이어
# 붙이므로, 공백을 가질 수 있는 값은 모두 이중 인용해서 넘긴다.
if ($Detached) {
    function Quote([string]$value) { '"' + $value + '"' }
    $forward = @("-ExecutionPolicy", "Bypass", "-NonInteractive", "-File", (Quote $PSCommandPath), "-Stage", $Stage)
    foreach ($name in @("RawRoot", "Config", "Plan", "Output", "Remote")) {
        $value = Get-Variable -Name $name -ValueOnly -ErrorAction SilentlyContinue
        if ($value) { $forward += @("-$name", (Quote $value)) }
    }
    $forward += @("-LimitPerModality", "$LimitPerModality")
    if ($TrustPlan) { $forward += "-TrustPlan" }
    if ($KeepDisplay) { $forward += "-KeepDisplay" }
    $child = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $forward -WindowStyle Hidden -PassThru
    Write-Host "$Stage 를 분리 실행으로 시작했다. PID $($child.Id)" -ForegroundColor Cyan
    if ($Output) { Write-Host "진행 상황: $Output-pipeline\pipeline.status" -ForegroundColor Cyan }
    exit 0
}

if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "pwsh(PowerShell 7) 필요. Windows PowerShell 5.1은 BOM 없는 UTF-8 한글을 깨뜨린다."
    exit 2
}

Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "가상환경이 없다. README의 설치 절차를 먼저 수행한다: $python"
    exit 2
}

if (-not $Output) { Write-Error "-Output 은 모든 단계에서 필요하다."; exit 2 }
# 래퍼 부기(status·로그)는 데이터 output 안이 아니라 형제 <Output>-pipeline 에 둔다.
# generate 는 빈 output 을 요구하므로, output 을 미리 만들거나 그 안에 로그를 쓰면
# "Output directory is not empty" 로 즉시 실패한다. output 자체는 각 툴이 만든다.
$pipeDir = "$Output-pipeline"
New-Item -ItemType Directory -Force -Path $pipeDir | Out-Null
$logDir = Join-Path $pipeDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "$($Stage)_$stamp.log"
$status = Join-Path $pipeDir "pipeline.status"

# smoke 가 측정한 worker_peak_rss_bytes 를 config 에 넣어 <config>.measured.json 을 만든다.
function Write-MeasuredConfig([string]$configPath, [string]$smokeOutput) {
    $summary = Join-Path $smokeOutput "generation_summary.json"
    if (-not (Test-Path $summary)) { throw "smoke summary 없음: $summary" }
    $measured = Join-Path (Split-Path $configPath -Parent) (
        [IO.Path]::GetFileNameWithoutExtension($configPath) + ".measured.json")
    & $python -X utf8 -c @"
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
summ = json.load(open(sys.argv[2], encoding='utf-8-sig'))
rss = int(summ.get('worker_peak_rss_bytes_observed') or summ.get('aggregate_peak_rss_bytes_observed') or 0)
if rss <= 0:
    raise SystemExit('measured RSS not found in summary')
cfg['worker_peak_rss_bytes'] = rss
json.dump(cfg, open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False, indent=2, sort_keys=True)
print(f'measured worker_peak_rss_bytes={rss} -> {sys.argv[3]}')
"@ $configPath $summary $measured 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "measured config 작성 실패" }
    return $measured
}

function Invoke-Upload {
    # 산출물 순서: 매니페스트·summary(작음, 감사 먼저) -> augmentation zip -> 이미지/라벨 트리.
    "[upload] 1/3 매니페스트·summary·로그" | Tee-Object -FilePath $log -Append
    & rclone copy $Output $Remote `
        --include "manifests/**" --include "*.json" --include "logs/**" `
        --transfers 8 --retries 5 --stats 30s --stats-one-line `
        --log-level INFO --log-file $log
    if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }

    $zips = Get-ChildItem $Output -Filter "*.zip" -ErrorAction SilentlyContinue
    if ($zips) {
        "[upload] 2/3 augmentation zip: $($zips.Count)개" | Tee-Object -FilePath $log -Append
        foreach ($zip in $zips) {
            $gb = [math]::Round($zip.Length / 1GB, 3)
            "[upload] 시작: $($zip.Name) ($gb GB)" | Tee-Object -FilePath $log -Append
            & rclone copyto $zip.FullName "$Remote/$($zip.Name)" --drive-chunk-size 128M --transfers 1 `
                --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
                --log-level INFO --log-file $log
            if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
        }
    }

    "[upload] 3/3 이미지·라벨 트리 (CT·RGB)" | Tee-Object -FilePath $log -Append
    & rclone copy $Output $Remote `
        --include "CT/**" --include "RGB/**" `
        --transfers 8 --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
        --log-level INFO --log-file $log
    return $LASTEXITCODE
}

switch ($Stage) {
    "plan" {
        if (-not $RawRoot -or -not $Config) { Write-Error "-RawRoot 와 -Config 필요"; exit 2 }
        $cliArgs = @("plan", "--raw-root", $RawRoot, "--config", $Config, "--output", $Output)
    }
    "smoke" {
        if (-not $RawRoot -or -not $Config -or -not $Plan) { Write-Error "-RawRoot·-Config·-Plan 필요"; exit 2 }
        $cliArgs = @("generate", "--raw-root", $RawRoot, "--config", $Config, "--plan", $Plan,
            "--output", $Output, "--limit-per-modality", "$LimitPerModality")
        if ($TrustPlan) { $cliArgs += "--trust-plan" }
    }
    "generate" {
        if (-not $RawRoot -or -not $Config -or -not $Plan) { Write-Error "-RawRoot·-Config·-Plan 필요"; exit 2 }
        $cliArgs = @("generate", "--raw-root", $RawRoot, "--config", $Config, "--plan", $Plan, "--output", $Output)
        if ($TrustPlan) { $cliArgs += "--trust-plan" }
    }
    "resume" {
        if (-not $RawRoot -or -not $Config -or -not $Plan) { Write-Error "-RawRoot·-Config·-Plan 필요"; exit 2 }
        $cliArgs = @("generate", "--raw-root", $RawRoot, "--config", $Config, "--plan", $Plan, "--output", $Output, "--resume")
        if ($TrustPlan) { $cliArgs += "--trust-plan" }
    }
    "verify" {
        $cliArgs = @("verify", "--output", $Output)
    }
    "upload" {
        if (-not $Remote) { Write-Error "-Remote 필요"; exit 2 }
        $cliArgs = $null
    }
}

Add-Type -Namespace Win32 -Name PipelinePower -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS       = [uint32]2147483648
$ES_SYSTEM_REQUIRED  = [uint32]1
$ES_DISPLAY_REQUIRED = [uint32]2
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if ($KeepDisplay) { $flags = $flags -bor $ES_DISPLAY_REQUIRED }

$start = Get-Date
"started $Stage $($start.ToString('yyyy-MM-dd HH:mm:ss')) pid=$PID" | Add-Content -Path $status
"[run_pipeline] $Stage / config $Config" | Tee-Object -FilePath $log -Append

$exit = 1
try {
    [Win32.PipelinePower]::SetThreadExecutionState($flags) | Out-Null
    if ($Stage -eq "upload") {
        $exit = Invoke-Upload
    }
    else {
        & $python -X utf8 -u -m "quality_fail_augment.cli" @cliArgs *>&1 | Tee-Object -FilePath $log -Append
        $exit = $LASTEXITCODE
        # generate 가 사람 QA 게이트에서 멈춘 경우는 실패가 아니라 대기다.
        if ($Stage -eq "generate" -and $exit -ne 0 -and (Select-String -Path $log -Pattern "Visual QA approval pending" -Quiet)) {
            $exit = 10
        }
        # smoke 성공 시 측정 config 를 만든다.
        if ($Stage -eq "smoke" -and $exit -eq 0) {
            $measured = Write-MeasuredConfig $Config $Output
            "[run_pipeline] 다음: -Stage plan -Config `"$measured`" 로 재-plan 후 generate" | Tee-Object -FilePath $log -Append
        }
    }
}
catch {
    $_ | Out-String | Tee-Object -FilePath $log -Append
    $exit = 1
}
finally {
    [Win32.PipelinePower]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    $end = Get-Date
    $duration = $end - $start
    "finished $Stage $($end.ToString('yyyy-MM-dd HH:mm:ss')) elapsed=$($duration.ToString('hh\:mm\:ss')) exit=$exit" |
        Add-Content -Path $status
}

if ($Stage -eq "generate" -and $exit -eq 10) {
    Write-Host "generate 완료 후 Visual QA 게이트 대기." -ForegroundColor Yellow
    Write-Host "사람이 검토: $Output\manifests\fail_visual_qa.csv (510장, reviewer·approved 채우기)" -ForegroundColor Yellow
    Write-Host "채운 뒤: -Stage resume 실행" -ForegroundColor Yellow
    exit 10
}
if ($exit -ne 0) {
    Write-Host "$Stage 실패 (exit=$exit). 로그: $log" -ForegroundColor Red
    exit $exit
}
Write-Host "$Stage 완료. 로그: $log" -ForegroundColor Green
exit 0
