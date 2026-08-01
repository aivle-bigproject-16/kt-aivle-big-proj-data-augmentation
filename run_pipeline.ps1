<#
  run_pipeline.ps1
  quality-fail-augment 파이프라인의 단일 실행 진입점이다.

  ## 스테이지

      plan     결정론적 plan 생성(병렬 scan). generation_plan.csv 와 scan_cache.csv 를
               만든다. -ReuseScan 으로 앞선 plan 의 scan_cache.csv 를 주면 원본이 그대로인
               쌍의 검증(약 60분)을 건너뛴다.
      smoke    소규모(기본 100장/모달리티) 생성으로 worker_peak_rss_bytes 를 측정하고,
               측정값을 넣은 <config>.measured.json 을 만든다. 이 키는 plan 해시 재료에서
               빠져 있으므로(planner.PERFORMANCE_ONLY_KEYS), 측정 config 로 바꿔도 이미
               승인된 plan 이 그대로 유효하다. 재-plan 이 필요 없다.
      generate 확정된 plan 으로 전체 생성하고 자동 검증 후 최종 ZIP 과 summary 를 만든다.
      resume   중단된 generate 작업을 기존 output 에서 이어서 실행한다.
      verify   생성된 데이터셋을 재검증한다.
      upload   생성 산출물을 rclone 으로 원격(gdrive 등)에 올린다. CT·RGB 트리는 zip
               하나로 묶어서 보낸다. -Remote 를 생략하면 기본 목적지
               gdrive:AIVLE_BigProject/data_augmentation/<Output 폴더명> 을 쓴다.
               -RawTree 를 주면 예전처럼 파일 단위로 올린다(수십 배 느리다).

  v1.8에는 사람 visual QA 게이트가 없다. generate 가 자동 검증까지 통과하면 최종
  산출물을 바로 만들며, resume 은 중단 복구에만 사용한다.

  ## 반드시 pwsh(PowerShell 7)로 실행한다

  Windows PowerShell 5.1은 BOM 없는 UTF-8 파일을 ANSI로 읽어 한글 경로를 깨뜨린다.

  ## 장시간 실행은 분리(-Detached)한다

  plan·smoke·generate 는 길게는 여러 시간이 걸린다. 에이전트 세션의 자식 프로세스는
  툴 호출이 끝나면 정리되므로, -Detached 로 스스로를 독립 프로세스로 다시 띄운다.

  ## 진행 상황

  단계마다 <Output>\pipeline.status 에 시작과 종료를 기록한다. 마지막 줄이
  finished ... exit=0 이면 정상 완료다.

  ## 경로 설정(.env)

  -RawRoot·-Config·-Plan·-Output 을 생략하면 저장소 루트의 `.env` 값을 쓴다.
  `.env.example` 을 `.env` 로 복사해 자기 머신 경로로 고쳐두면 아래처럼 단계 이름만
  주고 실행할 수 있다. 인자를 명시하면 그쪽이 항상 이긴다.

      QFA_RAW_ROOT    -RawRoot
      QFA_CONFIG      -Config
      QFA_PLAN_CSV    -Plan   (없으면 QFA_PLAN_DIR\manifests\generation_plan.csv)
      QFA_PLAN_DIR    -Output (plan 단계)
      QFA_SMOKE_DIR   -Output (smoke 단계)
      QFA_OUTPUT_DIR  -Output (generate·resume·verify·upload 단계)

  ## 사용 예

      pwsh -File .\run_pipeline.ps1 -Stage plan     -Detached
      pwsh -File .\run_pipeline.ps1 -Stage smoke    -TrustPlan -Detached
      pwsh -File .\run_pipeline.ps1 -Stage generate -TrustPlan -Detached
      pwsh -File .\run_pipeline.ps1 -Stage resume   -TrustPlan -Detached
      pwsh -File .\run_pipeline.ps1 -Stage verify
      pwsh -File .\run_pipeline.ps1 -Stage upload   -Detached

  경로를 직접 주고 싶을 때:

      pwsh -File .\run_pipeline.ps1 -Stage plan -RawRoot "..." -Config .\config.40k.json -Output "D:\qf_plan" -Detached

  계획 재작성이 필요할 때(quota·seed 등 계획에 영향을 주는 config 변경):

      pwsh -File .\run_pipeline.ps1 -Stage plan -Config .\config.40k.v2.json `
        -ReuseScan "D:\qf_plan\manifests\scan_cache.csv" -Output "D:\qf_plan_v2" -Detached
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
    [string]$DropCases,
    [string]$ReuseScan,
    [switch]$TrustPlan,
    [switch]$RawTree,
    [switch]$KeepDisplay,
    [switch]$Detached
)

# upload 기본 목적지. -Remote 를 생략하면 이 아래에 <Output 폴더명> 으로 올린다.
$DefaultRemoteRoot = "gdrive:AIVLE_BigProject/data_augmentation"

$ErrorActionPreference = "Stop"

# 버전 확인을 가장 먼저 한다. 아래 .env 읽기와 인자 전달이 모두 한글 경로를 다루므로,
# 5.1에서는 분리 실행으로 넘어가기 전에 멈춰야 한다.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "pwsh(PowerShell 7) 필요. Windows PowerShell 5.1은 BOM 없는 UTF-8 한글을 깨뜨린다."
    exit 2
}

# 생략된 경로 인자는 저장소 루트 .env 에서 채운다. 명시한 인자가 항상 우선한다.
# 값 형식은 src/quality_fail_augment/settings.py 와 같다.
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path -LiteralPath $envFile) {
    $envValues = @{}
    foreach ($line in (Get-Content -LiteralPath $envFile -Encoding UTF8)) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $split = $trimmed.IndexOf("=")
        if ($split -lt 1) { continue }
        $key = $trimmed.Substring(0, $split).Trim()
        $value = $trimmed.Substring($split + 1).Trim()
        if ($value.Length -ge 2 -and $value[0] -eq $value[-1] -and ($value[0] -eq '"' -or $value[0] -eq "'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $envValues[$key] = $value
    }
    if (-not $RawRoot -and $envValues["QFA_RAW_ROOT"]) { $RawRoot = $envValues["QFA_RAW_ROOT"] }
    if (-not $Config -and $envValues["QFA_CONFIG"]) { $Config = $envValues["QFA_CONFIG"] }
    if (-not $Plan) {
        if ($envValues["QFA_PLAN_CSV"]) {
            $Plan = $envValues["QFA_PLAN_CSV"]
        } elseif ($envValues["QFA_PLAN_DIR"]) {
            $Plan = Join-Path $envValues["QFA_PLAN_DIR"] "manifests\generation_plan.csv"
        }
    }
    if (-not $Output) {
        # 단계마다 기본 출력 폴더가 다르다. plan 은 계획, smoke 는 시험 생성,
        # 나머지는 본 생성 산출물을 가리킨다.
        $outputKey = switch ($Stage) {
            "plan"  { "QFA_PLAN_DIR" }
            "smoke" { "QFA_SMOKE_DIR" }
            default { "QFA_OUTPUT_DIR" }
        }
        if ($envValues[$outputKey]) { $Output = $envValues[$outputKey] }
    }
}

# 긴 단계는 호출 세션의 자식이 되면 안 된다. -Detached 로 독립 프로세스로 다시 띄운다.
# Start-Process -ArgumentList 는 공백을 포함한 배열 원소를 인용하지 않고 공백으로 이어
# 붙이므로, 공백을 가질 수 있는 값은 모두 이중 인용해서 넘긴다.
if ($Detached) {
    function Quote([string]$value) { '"' + $value + '"' }
    $forward = @("-ExecutionPolicy", "Bypass", "-NonInteractive", "-File", (Quote $PSCommandPath), "-Stage", $Stage)
    foreach ($name in @("RawRoot", "Config", "Plan", "Output", "Remote", "DropCases", "ReuseScan")) {
        $value = Get-Variable -Name $name -ValueOnly -ErrorAction SilentlyContinue
        if ($value) { $forward += @("-$name", (Quote $value)) }
    }
    $forward += @("-LimitPerModality", "$LimitPerModality")
    if ($TrustPlan) { $forward += "-TrustPlan" }
    if ($RawTree) { $forward += "-RawTree" }
    if ($KeepDisplay) { $forward += "-KeepDisplay" }
    $child = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $forward -WindowStyle Hidden -PassThru
    Write-Host "$Stage 를 분리 실행으로 시작했다. PID $($child.Id)" -ForegroundColor Cyan
    if ($Output) { Write-Host "진행 상황: $Output-pipeline\pipeline.status" -ForegroundColor Cyan }
    exit 0
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
    $note = & $python -X utf8 -c @"
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
summ = json.load(open(sys.argv[2], encoding='utf-8-sig'))
rss = int(summ.get('worker_peak_rss_bytes_observed') or summ.get('aggregate_peak_rss_bytes_observed') or 0)
if rss <= 0:
    raise SystemExit('measured RSS not found in summary')
cfg['worker_peak_rss_bytes'] = rss
json.dump(cfg, open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False, indent=2, sort_keys=True)
print(f'measured worker_peak_rss_bytes={rss} -> {sys.argv[3]}')
"@ $configPath $summary $measured 2>&1
    $exit = $LASTEXITCODE
    # 위 출력을 파이프라인으로 흘리면 이 함수의 반환값이 $measured 하나가 아니라 그
    # 출력까지 담은 배열이 된다. 호출부는 그 값을 -Config 경로로 안내하므로, 문구는
    # 호스트와 로그에만 직접 쓴다.
    $note | ForEach-Object { Write-Host $_; Add-Content -Path $log -Value $_ }
    if ($exit -ne 0) { throw "measured config 작성 실패" }
    return $measured
}

# Invoke-Upload 안의 안내 문구는 파이프라인으로 흘리면 안 된다. 함수의 반환값이
# rclone 종료 코드 하나가 아니라 그 문구들까지 담은 배열이 되고, 아래 "$exit -ne 0"
# 판정과 pipeline.status 의 exit= 값이 둘 다 망가진다.
function Write-UploadNote([string]$text) {
    Write-Host $text
    Add-Content -Path $log -Value $text
}

function Invoke-Upload {
    # 산출물 순서: 매니페스트·summary(작음, 감사 먼저) -> augmentation zip -> 이미지/라벨.
    # 패턴 앞의 / 는 전송 루트 기준을 뜻한다. 이게 없으면 rclone 은 모든 깊이에서
    # 매칭하므로 --include "*.json" 이 CT/RGB 트리 안의 라벨·이력 JSON 44,000 개까지
    # 끌어와 파일 단위로 올린다. 3/3 에서 트리를 묶는 의미가 사라진다.
    Write-UploadNote "[upload] 1/3 매니페스트·summary·로그"
    & rclone copy $Output $Remote `
        --include "/manifests/**" --include "/*.json" --include "/logs/**" `
        --transfers 8 --retries 5 --stats 30s --stats-one-line `
        --log-level INFO --log-file $log
    if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }

    $zips = Get-ChildItem $Output -Filter "*.zip" -ErrorAction SilentlyContinue
    if ($zips) {
        Write-UploadNote "[upload] 2/3 augmentation zip: $($zips.Count)개"
        foreach ($zip in $zips) {
            $gb = [math]::Round($zip.Length / 1GB, 3)
            Write-UploadNote "[upload] 시작: $($zip.Name) ($gb GB)"
            & rclone copyto $zip.FullName "$Remote/$($zip.Name)" --drive-chunk-size 128M --transfers 1 `
                --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
                --log-level INFO --log-file $log
            if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
        }
    }

    $tree = @("CT", "RGB") | Where-Object { Test-Path (Join-Path $Output $_) }
    if (-not $tree) {
        Write-UploadNote "[upload] 3/3 CT·RGB 트리 없음, 건너뜀"
        return 0
    }

    if ($RawTree) {
        # 개별 파일을 Drive 에서 그대로 열람해야 할 때만 쓴다. 아래 아카이브 경로보다
        # 수십 배 느리다.
        Write-UploadNote "[upload] 3/3 이미지·라벨 트리 (CT·RGB, 파일 단위)"
        & rclone copy $Output $Remote `
            --include "/CT/**" --include "/RGB/**" `
            --transfers 8 --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
            --log-level INFO --log-file $log
        return $LASTEXITCODE
    }

    # Google Drive 는 파일마다 API 왕복이 필요하다. 40,000 장 + 라벨 = 약 84,000 개를
    # 파일 단위로 올리면 용량(1 GB 미만)과 무관하게 몇 시간이 걸린다. 트리를 zip 하나로
    # 묶어 큰 파일 한 개로 올리면 같은 데이터가 10 분대에 끝난다.
    $archive = Join-Path $pipeDir ((Split-Path $Output -Leaf) + "_images.zip")
    Write-UploadNote "[upload] 3/3 이미지·라벨 아카이브 생성: $archive"
    if (Test-Path $archive) { Remove-Item $archive -Force }
    # bsdtar(Windows 기본 tar.exe). -a 로 확장자에서 zip 포맷을 고른다. JPG 는 이미
    # 압축돼 있어 줄지 않지만 JSON 라벨이 크게 줄고, 무엇보다 파일 수가 1 개가 된다.
    & tar.exe -a -c -f $archive -C $Output @tree
    if ($LASTEXITCODE -ne 0) {
        Write-UploadNote "[upload] 아카이브 생성 실패 (exit=$LASTEXITCODE)"
        return $LASTEXITCODE
    }
    $archiveItem = Get-Item $archive
    $gb = [math]::Round($archiveItem.Length / 1GB, 3)
    Write-UploadNote "[upload] 아카이브 완료: $($archiveItem.Name) ($gb GB) -> 전송 시작"
    & rclone copyto $archive "$Remote/$($archiveItem.Name)" --drive-chunk-size 128M --transfers 1 `
        --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
        --log-level INFO --log-file $log
    return $LASTEXITCODE
}

switch ($Stage) {
    "plan" {
        if (-not $RawRoot -or -not $Config) { Write-Error "-RawRoot 와 -Config 필요"; exit 2 }
        $cliArgs = @("plan", "--raw-root", $RawRoot, "--config", $Config, "--output", $Output)
        # 앞선 plan 의 scan_cache.csv 를 재사용해 쌍 검증(약 60분)을 건너뛴다.
        if ($ReuseScan) { $cliArgs += @("--reuse-scan", $ReuseScan) }
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
        # 파라미터를 변경한 특정 case만 다시 만들 때 사용한다.
        if ($DropCases) { $cliArgs += @("--drop-cases", $DropCases) }
    }
    "verify" {
        $cliArgs = @("verify", "--output", $Output)
    }
    "upload" {
        if (-not $Remote) {
            $Remote = "$DefaultRemoteRoot/$(Split-Path $Output -Leaf)"
            "[run_pipeline] -Remote 생략, 기본 목적지 사용: $Remote" | Tee-Object -FilePath $log -Append
        }
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
        # smoke 성공 시 측정 config 를 만든다.
        if ($Stage -eq "smoke" -and $exit -eq 0) {
            $measured = Write-MeasuredConfig $Config $Output
            # worker_peak_rss_bytes 는 PERFORMANCE_ONLY_KEYS 라 plan 해시를 바꾸지 않는다.
            # 재-plan 없이 기존 plan 그대로 generate 로 넘어간다.
            "[run_pipeline] 다음: -Stage generate -Config `"$measured`" 로 기존 plan 그대로 생성" | Tee-Object -FilePath $log -Append
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

if ($exit -ne 0) {
    Write-Host "$Stage 실패 (exit=$exit). 로그: $log" -ForegroundColor Red
    exit $exit
}
Write-Host "$Stage 완료. 로그: $log" -ForegroundColor Green
exit 0
