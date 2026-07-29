# quality-fail-augment 1.8

`CT RGB 품질 PASS FAIL 독립 데이터셋 생성 계획서 v1.8`의 실행 코드다.

원본 배터리 이미지에서 PASS 샘플과 **인위적으로 촬영 결함을 주입한 FAIL 샘플**을 만들어,
품질 판정 모델이 학습할 40,000장 규모의 데이터셋을 결정론적으로 생성한다.

## 출력 계약

- CT 이미지 20,000장: PASS 18,000 / FAIL 2,000
- RGB 이미지 20,000장: PASS 18,000 / FAIL 2,000
- CT 대응 라벨 JSON 20,000개
- RGB 대응 라벨 JSON 20,000개
- CT FAIL 증강 이력 JSON 2,000개
- RGB FAIL 증강 이력 JSON 2,000개
- PASS에는 증강 이력 JSON을 생성하지 않는다.
- 각 모달리티 test는 1,000장(PASS 900 / FAIL 100)이다.

원본 이미지와 JSON은 읽기 전용이다. 출력 대응 라벨 JSON만 신규 battery/image
ID, 파일명, 출력 크기, `quality_class`, 변환된 polygon 좌표를 갖는다.

## 파이프라인 전체 흐름

이 저장소는 다음 단계를 순서대로 실행해 최종 데이터셋을 만드는 구조다. 각 단계는
앞 단계의 산출물을 해시로 대조하며, 하나라도 어긋나면 다음 단계가 실행을 거부한다.

```text
[1] audit-raw   원본을 읽기 전용으로 훑어 이 코드가 다룰 수 있는 데이터인지 확인
       ↓        산출: 감사 리포트 (원본은 건드리지 않는다)
[2] plan        원본 전량 스캔 → 어떤 원본이 어떤 출력이 될지 전부 확정
       ↓        산출: generation_plan.csv, reserve_sources.csv, raw fingerprint,
       ↓              scan_cache.csv (다음 plan 에서 --reuse-scan 으로 재사용)
[3] smoke       소량 생성으로 워커 메모리 실측 → 병렬 워커 수 결정용 config 확보
       ↓        산출: generation_summary.json 의 worker_peak_rss_bytes_observed
       ↓        측정값은 성능 전용 키라 [2] 의 plan 을 무효화하지 않는다
[4] generate    plan 대로 40,000장 생성 → 자동 검증 → 최종 산출물 생성
       ↓        산출: 이미지·라벨·이력 전량, manifests/,
       ↓              generation_summary.json, augmentation_json_4k_v1.8.zip
[5] verify      생성된 데이터셋 재검증
[6] upload      원격 저장소로 전송
```

v1.8에는 사람 visual QA 게이트가 없다. `generate`가 자동 검증까지 통과하면
`generation_summary.json`과 ZIP을 포함한 최종 산출물을 바로 만든다. 실행이 중단된 경우에만
`resume`으로 이어서 생성한다.

### 단계별 결정론 보장

- `plan`이 원본 전체의 fingerprint를 기록한다. `generate`는 실행 시작 시 원본을 다시 스캔해
  fingerprint가 같은지 대조하고, 다르면 생성을 거부한다.
- config 파일이 바뀌면 `config_sha256`이 바뀌고, 승인된 plan과 불일치하므로 역시 거부한다.
  단 산출물에 영향을 주지 않는 **성능 전용 키는 해시 재료에서 제외**한다
  (`planner.PERFORMANCE_ONLY_KEYS`). 그래서 smoke가 측정한 `worker_peak_rss_bytes`를
  config에 넣어도 승인된 plan이 그대로 유효하다. 목록에 없는 키는 전부 해시에 남으므로,
  새 키를 추가하면 기본적으로 plan이 무효화되는 안전한 방향으로 동작한다.
- 샘플마다 `item_seed`가 고정이라, 같은 코드·같은 plan이면 같은 이미지가 나온다.

### scan 캐시 — plan 재실행 비용 줄이기

v1.8의 scan cache v3는 RGB glare와 CT alignment 원본의 유효 outline 적격성을
다시 판단할 수 있도록 `has_battery_outline`을 함께 저장한다. 기존 v1/v2 캐시는 판정
의미가 다르므로 최초 한 번은
`-ReuseScan` 없이 전체 원본을 다시 스캔해야 한다. 이 실행이 완료되면
`<Output>\manifests\scan_cache.csv`가 생성되고, 다음 plan부터 그 파일을 `--reuse-scan`으로
지정한다.

`run_pipeline.ps1`은 PowerShell 7 이상이 필요하다. `pwsh`가 없는 Windows PowerShell
5에서는 아래처럼 가상환경의 Python CLI를 직접 실행한다.

```powershell
# 최초 1회: 기존 캐시를 지정하지 않고 v3 캐시 생성
& ".\.venv\Scripts\python.exe" -m quality_fail_augment.cli plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --output "E:\quality_fail_40k_plan_v1.8_cache_v3"

# 다음 plan부터: 최초 실행에서 생성한 v3 캐시 재사용
& ".\.venv\Scripts\python.exe" -m quality_fail_augment.cli plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --reuse-scan "E:\quality_fail_40k_plan_v1.8_cache_v3\manifests\scan_cache.csv" `
  --output "E:\quality_fail_40k_plan_v1.8_next"
```

첫 실행에는 시간이 오래 걸리는 것이 정상이다. 두 번째 실행부터는 변경되지 않은
image–JSON pair를 v3 캐시에서 복원한다. 기존 v1/v2 캐시를 `--reuse-scan`으로 지정하면 스키마
불일치 오류가 발생하므로 사용하지 않는다. 직접 실행 로그는 각 출력 폴더의
`logs\plan.log`에도 저장된다. PowerShell 7이 설치된 환경에서는 아래 자동 파이프라인
절의 `pwsh -File .\run_pipeline.ps1 ...` 명령을 사용할 수 있다.

`plan`의 시간은 거의 전부 2단계인 쌍 검증이다. 원본 파일 탐색은 수분이지만, 쌍 276,170건을
각각 이미지 전량 디코드 + 이미지·JSON SHA-256 + 픽셀 해시로 검증하므로 저장장치와 파일 크기에 따라
최초 실행은 수시간 이상 걸릴 수 있다.

`plan`과 `audit-raw`는 그 결과를 `manifests/scan_cache.csv`로 남긴다. 다음 실행에
`--reuse-scan`으로 그 파일(또는 그것을 담은 디렉터리)을 주면, 이미지와 JSON의 크기·수정시각이
그대로인 쌍은 캐시에서 가져오고 나머지만 검증한다.

```powershell
quality-fail-augment plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --reuse-scan "E:\quality_fail_40k_plan_v1.8_cache_v3\manifests\scan_cache.csv" `
  --output "E:\quality_fail_40k_plan_v1.8_next"
```

캐시가 온전히 적중하면 plan은 최초 수시간에서 몇 분 수준으로 줄고, `generation_plan.csv`는
전체 재스캔으로 만든 것과 **바이트 단위로 동일**하다(`tests/test_scan_cache.py`).

무효화 규칙:

| 바뀐 것 | 캐시 유효 |
|---|---|
| quota·seed·target 등 계획용 config 키 | 유효. scan은 이 값들을 읽지 않는다 |
| `ct_porosity_threshold` | 유효. 다공성 비율 원값을 캐시하므로 임계값만 다시 적용한다 |
| 원본 파일 추가·삭제·수정 | 해당 쌍만 무효. 1단계 탐색과 크기·수정시각 대조로 잡는다 |
| `_validate_pair`가 쓰는 검증 로직 | 전부 무효. `planner.SCAN_CACHE_VERSION`을 올려야 한다 |

신선도를 크기·수정시각으로만 판정하는 것은 의도적이다. SHA-256 재계산이 바로 이 캐시가
없애려는 비용이기 때문이다. 대신 plan에 실제로 선택된 40k+reserve 소스는 `generate`가
`--trust-plan`이어도 SHA-256으로 다시 pin 검증하므로(`_validate_plan`), 실제 사용되는
소스의 무결성은 그대로 보장된다. 미선택 소스까지 전량 재해싱하고 싶으면
`--reuse-scan` 없이 돌리면 된다.

## 코드 구조

이미지 한 장의 전처리는
`src/quality_fail_augment/preprocessing/stages.py`에 다음 순서로 분리되어 있다.

```text
prepare_source()
  원본 이미지·JSON 1회 로드
  CT ROI crop
  배터리/결함 mask 생성
        ↓
apply_quality_transform()
  PASS: failure case를 적용하지 않음
  FAIL: 배정된 failure case 1개 적용
  이미지 변환과 affine 행렬을 함께 누적
        ↓
finalize_sample()
  최종 resize
  동일 affine 행렬로 outline/defect polygon 좌표 변환
        ↓
generator._make_one()
  이미지·라벨·FAIL 이력 직렬화
  staging 검증 및 최종 경로 commit
```

역할별 파일:

- `planner.py`: 원본 탐색, image–JSON pairing, 감사, source 선택, plan 생성. 쌍별
  검증(이미지 디코드·해시·porosity)은 config `jobs` 수만큼 프로세스로 병렬 처리하고,
  픽셀 중복 제거와 계통 오류 판정 등 순서 의존 병합은 정렬 순서대로 직렬 수행해
  raw fingerprint를 병렬·직렬 동일하게 유지한다
- `preprocessing/stages.py`: 한 sample의 순차 전처리와 좌표 동기화
- `augment.py`: CT/RGB failure case의 실제 영상 변환과 **자동 품질 게이트**
- `geometry.py`: ROI, polygon parsing/clipping, affine 좌표 변환
- `generator.py`: 병렬 실행, 재시도, 파일 commit, manifest와 최종 자동 검증
- `cli.py`: `audit-raw`, `plan`, `generate`, `verify` 명령 진입점

## 품질 검사

각 이미지는 `augment.py`의 `_quality_gate()`에서 증강 case별 수치 조건을 검사한다.
조건을 통과하지 못하면 같은 case로 최대 8회 재시도한 뒤 reserve source로 교체한다.
전체 생성이 끝나면 `generator.py`가 수량, 이미지·JSON 쌍, 해시, 매니페스트와 출력 구조를
다시 검증한다.

## v1.8 검증 방식

v1.8은 사람 visual QA 게이트와 `fail_visual_qa.csv` 승인 절차를 사용하지 않는다.
각 증강 결과는 생성 중 자동 품질 게이트를 통과해야 하며, 전체 생성 후에는 이미지·JSON 쌍,
수량, 해시, 매니페스트와 출력 구조를 자동 검증한다. 자동 검증이 성공하면 최종 ZIP과
`generation_summary.json`을 즉시 생성한다.

`resume`은 QA 승인용 단계가 아니라 중단된 생성 작업을 이어가기 위한 복구 단계다.
특정 failure case의 파라미터를 바꾼 뒤 그 case만 다시 생성해야 할 때는
`--resume --drop-cases <case 목록>`을 사용할 수 있다.

## 설치

```powershell
cd "quality_fail_augment_code_v1.8"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
quality-fail-augment --version
```

버전 출력은 정확히 `1.8`이어야 한다.

## 실행

먼저 실제 raw-root를 읽기 전용으로 감사한다.

```powershell
quality-fail-augment audit-raw `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --output "E:\quality_fail_raw_audit_v1.8"
```

결정론적 source, output slot, main/test, failure case를 고정한다.

```powershell
quality-fail-augment plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --output "E:\quality_fail_40k_plan_v1.8"
```

소규모 시험 생성:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_smoke_v1.8" `
  --limit-per-modality 100
```

첫 smoke는 `worker_peak_rss_bytes`가 없으므로 안전하게 worker 1개로 실행된다.
`generation_summary.json`의 `worker_peak_rss_bytes_observed`를
`config.40k.json`의 `worker_peak_rss_bytes`로 추가한 뒤 plan을 다시 생성하면,
이 측정값과 실행 직전 available memory의 70%를 사용해 병렬 worker 수를
결정한다. 측정값 없이 `jobs=8`만 지정해도 임의의 메모리 추정치로 병렬화하지
않는다.

전체 생성:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.8"
```

전체 파일 생성과 자동 검증이 성공하면 최종 summary와 ZIP까지 바로 생성된다.

재개:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.8" `
  --resume
```

`--resume`은 manifest에 이미 기록된 `synthetic_id`를 건너뛰고 나머지만 생성한다.
따라서 QA 통과 후 재개하면 이미지를 다시 만들지 않고 검증과 릴리스 산출물 생성만 수행한다.

검증:

```powershell
quality-fail-augment verify --output "E:\quality_fail_40k_v1.8"
```

## 자동 파이프라인 (`run_pipeline.ps1`)

`run_pipeline.ps1`은 위의 개별 명령을 하나의 진입점으로 묶는다. pwsh(PowerShell 7)로
실행하며, 각 단계는 실행 유지(keep-awake)와 `-Detached` 분리 실행, `<Output>-pipeline\pipeline.status`
하트비트, 단계별 로그를 지원한다. 래퍼의 상태와 로그는 데이터 output 안이 아니라 형제
`<Output>-pipeline` 폴더에 쌓인다(generate 는 빈 output 을 요구하기 때문이다). 스테이지는 `plan`, `smoke`, `generate`, `resume`,
`verify`, `upload`이다.

```powershell
# 1) 병렬 scan 으로 plan 생성
pwsh -File .\run_pipeline.ps1 -Stage plan `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.json `
  -Output "E:\quality_fail_40k_plan_v1.8" -Detached

# 2) smoke 로 worker_peak_rss_bytes 측정 → config.40k.measured.json 자동 생성
pwsh -File .\run_pipeline.ps1 -Stage smoke `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.json `
  -Plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_smoke_v1.8" -Detached

# 3) 측정 config 로 full 생성하고 자동 검증 후 최종 ZIP까지 만든다.
#    측정값은 성능 전용 키라 plan 해시를 바꾸지 않으므로, 1) 의 plan 을 그대로 쓴다.
pwsh -File .\run_pipeline.ps1 -Stage generate `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.measured.json `
  -Plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_v1.8" -Detached

# 중단된 작업이 있을 때만 resume으로 이어서 생성
pwsh -File .\run_pipeline.ps1 -Stage resume `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.measured.json `
  -Plan "E:\quality_fail_40k_plan_v1.8\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_v1.8" -Detached

# 4) 원격(gdrive 등) 업로드
pwsh -File .\run_pipeline.ps1 -Stage upload `
  -Output "E:\quality_fail_40k_v1.8" -Remote "gdrive:quality_fail_40k_v1.8" -Detached
```

v1.8의 `generate`는 사람 QA 대기 없이 자동 검증과 최종 산출물 생성을 완료한다.
`upload`는 `rclone`이 설치되고 원격이 설정돼 있어야 하며,
매니페스트와 summary를 먼저 올린 뒤 augmentation ZIP과 이미지·라벨 트리를 올린다.

`generate`, `smoke`, `resume`은 시작할 때마다 plan 검증을 위해 전체 raw를 다시 스캔한다
(약 44분). plan 직후처럼 원본이 그대로임이 확실하면 `-TrustPlan`(툴의 `--trust-plan`)을
주어 이 전체 재스캔과 fingerprint 재대조를 건너뛸 수 있다. 이때도 plan이 쓰는
40k+reserve 소스는 각각 SHA-256으로 여전히 검증하므로 선택된 소스의 무결성은 보장된다.
smoke 기준으로 재스캔 포함 약 53분이 `-TrustPlan`에서는 약 3분으로 줄어든다.

### 운영 주의

생성이 도는 동안 `manifests/*.csv`를 열어 읽지 말 것. Windows에서 리더가 파일을 잡고
있으면 `os.replace` 경합을 유발한다. 진행 상황은 `pipeline.status`와 `logs/`로만 확인한다.

## 중단과 제외

- ambiguous `(raw_split, modality, stem)` pair는 전체 plan을 중단한다.
- image-only, JSON-only와 개별 손상/schema 오류는 감사 후 source만 제외한다.
- 같은 오류가 preflight 5건, 전체 scan 연속 20건 또는 누적 100건이면 중단한다. 단
  CT porosity 초과와 픽셀 동일 중복(`duplicate_pixel_hash`)은 의도된 정상 제외이므로
  이 계통 오류 카운터에서 면제한다.
- config hash, source hash 또는 raw fingerprint가 승인 plan과 다르면 생성하지 않는다.
- 생성 실패는 같은 case로 재시도하고 같은 modality의 reserve source로 교체한다.
- reserve까지 소진하면 전체 생성을 실패 처리한다.
- PASS는 2파일, FAIL은 3파일을 staging에서 검증한 뒤 manifest에 commit한다.

## 작업 기록

- `docs/작업노트_2026-07-24.md`: 첫 전량 실행에서 잡은 성능·안정성·품질 결함
- `docs/시각검수_결과_2026-07-27.md`: 과거 v1.6 시각검수 결과와 증강법 개선 근거
- `docs/작업노트_2026-07-28_visual_qa.md`: 과거 검수 결과를 반영한 증강법 수정 기록

## 테스트

```powershell
$env:PYTHONPATH = (Resolve-Path ".\src").Path
python -m unittest discover -s tests -v
```
