# quality-fail-augment 1.5

`CT RGB 품질 PASS FAIL 독립 데이터셋 생성 계획서 v1.5`의 실행 코드다.

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

이 저장소는 여섯 단계를 순서대로 통과해야 최종 데이터셋이 공개되는 구조다. 각 단계는
앞 단계의 산출물을 해시로 대조하며, 하나라도 어긋나면 다음 단계가 실행을 거부한다.

```text
[1] audit-raw   원본을 읽기 전용으로 훑어 이 코드가 다룰 수 있는 데이터인지 확인
       ↓        산출: 감사 리포트 (원본은 건드리지 않는다)
[2] plan        원본 전량 스캔 → 어떤 원본이 어떤 출력이 될지 전부 확정
       ↓        산출: generation_plan.csv, reserve_sources.csv, raw fingerprint
[3] smoke       소량 생성으로 워커 메모리 실측 → 병렬 워커 수 결정용 config 확보
       ↓        산출: generation_summary.json 의 worker_peak_rss_bytes_observed
[4] generate    plan 대로 40,000장 생성 → 자동 검증 → 사람 visual QA 게이트에서 정지
       ↓        산출: 이미지·라벨·이력 전량, manifests/, fail_visual_qa.csv
[5] resume      사람이 채운 QA CSV 를 검사 → 통과하면 최종 산출물 공개
       ↓        산출: generation_summary.json, augmentation_json_4k_v1.5.zip
[6] upload      원격 저장소로 전송
```

[4]에서 파일은 이미 전부 만들어지지만 **`generation_summary.json` 과 ZIP 은 나오지 않는다.**
사람 검수를 통과해야만 [5]가 그 둘을 만든다. 즉 "생성 완료"와 "릴리스 가능"은 다른 상태다.

### 단계별 결정론 보장

- `plan`이 원본 전체의 fingerprint를 기록한다. `generate`는 실행 시작 시 원본을 다시 스캔해
  fingerprint가 같은지 대조하고, 다르면 생성을 거부한다.
- config 파일이 바뀌면 `config_sha256`이 바뀌고, 승인된 plan과 불일치하므로 역시 거부한다.
  성능 전용 키(`worker_peak_rss_bytes` 등)를 추가해도 해시가 바뀌므로 plan을 다시 만들어야 한다.
- 샘플마다 `item_seed`가 고정이라, 같은 코드·같은 plan이면 같은 이미지가 나온다.

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
- `generator.py`: 병렬 실행, 재시도, 파일 commit, manifest와 **visual QA 게이트**
- `cli.py`: `audit-raw`, `plan`, `generate`, `verify` 명령 진입점

## 품질 검사는 2단계다

이 저장소에는 성격이 다른 검사가 두 겹으로 들어간다. 헷갈리면 안 된다.

| 구분 | 자동 품질 게이트 | 사람 visual QA 게이트 |
|---|---|---|
| 위치 | `augment.py` `_quality_gate()` | `generator.py` `_visual_qa_gate()` |
| 시점 | 샘플 1장을 만들 때마다 | 40,000장을 다 만든 뒤 1회 |
| 검사 대상 | 그 장의 픽셀 통계 | case별 표본 30장의 육안 인상 |
| 판정 기준 | 수치 임계값 (예: 엣지 손실 30~80%) | 사람이 "이게 FAIL로 보이는가" |
| 실패 시 | 같은 case로 최대 8회 재시도 → reserve source로 교체 | 해당 case 전체 재생성 |

**자동 게이트만으로는 부족하다.** 자동 게이트는 "머리카락 샘플점의 50% 이상이 배터리 마스크
안에 있는가", "좌우 절반의 평균 휘도 차가 전체 평균의 12% 이상인가" 같은 **수치 조건**만 본다.
수치는 통과했는데 사람 눈에는 정상 사진과 구별되지 않는 합성물이 얼마든지 나올 수 있다.

이 데이터셋은 하류 품질 판정 모델이 학습할 FAIL 라벨의 근거다. FAIL이라고 라벨링됐지만
실제로는 정상으로 보이는 이미지가 섞이면, 모델은 결함이 아니라 라벨 노이즈를 학습한다.
그래서 릴리스 전에 사람 눈을 한 번 통과시킨다.

## visual QA 게이트 — `fail_visual_qa.csv`를 왜 채워야 하는가

### 무엇을 하는 게이트인가

`generate`가 40,000장을 다 만들고 자동 검증까지 끝내면, 마지막으로 FAIL 샘플 중 **case별
30장**을 결정론적으로 뽑아 `manifests/fail_visual_qa.csv`를 만들고 **실행을 중단한다**
(종료 코드 10, 메시지 `Visual QA approval pending`).

- CT 8개 case × 30장 = 240장
- RGB 9개 case × 30장 = 270장
- 합계 **510장**
- 단 `ct_detector_calibration`은 하위 유형이 둘이라 ring 15장 + stripe 15장으로 나눠 뽑는다

표본은 `synthetic_id` 정렬 순 상위 30개로 고정이다. 무작위가 아니므로 같은 데이터셋이면
같은 510장이 뽑히고, 재생성해도 `synthetic_id`가 유지되면 표본도 그대로다.

### 검수자가 할 일

CSV의 각 행에 대해 해당 이미지를 보고 두 칸을 채운다.

| 칼럼 | 채울 값 |
|---|---|
| `reviewer` | 검수자 이름. 비어 있으면 게이트가 거부한다 |
| `approved` | `true` / `false` (`yes`/`no`, `1`/`0` 도 허용). 비어 있거나 다른 값이면 거부한다 |
| `reason` | 선택. `false`로 준 경우 왜 그렇게 판단했는지 남기면 원인 분석이 빨라진다 |

판정 기준은 **"이 이미지가 해당 failure case의 촬영 결함으로 보이는가"** 하나다.
예를 들어 `rgb_trigger_timing_failure`는 트리거가 늦거나 빨라 배터리가 프레임 밖으로
잘려나간 상황을 뜻하므로, 배터리가 프레임 안에 온전히 담겨 있으면 `false`다.

**행을 추가하거나 삭제하면 안 된다.** 게이트가 CSV의 `synthetic_id` 집합을 자기가 뽑은
표본과 대조하고, 하나라도 다르면 `Visual QA CSV sample IDs differ...`로 거부한다.
CSV는 UTF-8(BOM 포함)로 읽으므로 Excel에서 편집해 저장해도 무방하다.

### 통과 기준

**case별 승인율 95% 이상**(`visual_qa_min_approval_rate`). case당 30장이므로 실질적으로
**case당 반려 1장까지 허용**되고, 2장부터 그 case는 실패다.

승인율이 미달이면 `resume`이 다음 메시지로 중단한다.

```text
Visual QA approval below 95% for RGB/rgb_uneven_lighting: 63.33%;
adjust parameters and regenerate the entire case
```

이때 할 일은 **해당 case의 증강 파라미터를 고쳐 그 case 전체를 다시 만드는 것**이다.
개별 반려 샷만 교체하는 것은 허용되지 않는다. 반려가 나왔다는 건 그 case의 생성 로직
자체가 약하다는 뜻이고, 표본에 안 걸린 나머지 장도 같은 문제를 갖고 있기 때문이다.

### 배포와 회수

검수를 팀에 맡길 때는 다음 세 가지를 함께 넘긴다.

- `manifests/fail_visual_qa.csv` (빈 상태)
- 표본 510장의 이미지
- 위 판정 기준

검수자가 채운 CSV를 받아 `manifests/fail_visual_qa.csv` 자리에 덮어쓰고 `--resume`으로
재개한다. 게이트는 CSV가 이미 존재하면 새로 만들지 않고 그 내용을 검사한다.

## 설치

```powershell
cd "quality_fail_augment_code_v1.5"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
quality-fail-augment --version
```

버전 출력은 정확히 `1.5`여야 한다.

## 실행

먼저 실제 raw-root를 읽기 전용으로 감사한다.

```powershell
quality-fail-augment audit-raw `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --output "E:\quality_fail_raw_audit_v1.5"
```

결정론적 source, output slot, main/test, failure case를 고정한다.

```powershell
quality-fail-augment plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --output "E:\quality_fail_40k_plan_v1.5"
```

소규모 시험 생성:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_smoke_v1.5" `
  --limit-per-modality 100
```

첫 smoke는 `worker_peak_rss_bytes`가 없으므로 안전하게 worker 1개로 실행된다.
`generation_summary.json`의 `worker_peak_rss_bytes_observed`를
`config.40k.json`의 `worker_peak_rss_bytes`로 추가한 뒤 plan을 다시 생성하면,
이 측정값과 실행 직전 available memory의 70%를 사용해 병렬 worker 수를
결정한다. 측정값 없이 `jobs=8`만 지정해도 임의의 메모리 추정치로 병렬화하지
않는다. `--limit-per-modality`를 준 실행은 visual QA 게이트를 거치지 않는다.

전체 생성:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.5"
```

전체 파일 생성·자동 검증 뒤에는 `manifests/fail_visual_qa.csv`가 생성되고 첫
실행이 `Visual QA approval pending`으로 중단된다. 위 "visual QA 게이트" 절의 기준대로
510장을 검수해 CSV를 채운 뒤 재개한다.

재개:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.5" `
  --resume
```

`--resume`은 manifest에 이미 기록된 `synthetic_id`를 건너뛰고 나머지만 생성한다.
따라서 QA 통과 후 재개하면 이미지를 다시 만들지 않고 검증과 릴리스 산출물 생성만 수행한다.

검증:

```powershell
quality-fail-augment verify --output "E:\quality_fail_40k_v1.5"
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
  -Output "E:\quality_fail_40k_plan_v1.5" -Detached

# 2) smoke 로 worker_peak_rss_bytes 측정 → config.40k.measured.json 자동 생성
pwsh -File .\run_pipeline.ps1 -Stage smoke `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.json `
  -Plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_smoke_v1.5" -Detached

# 3) 측정 config 로 plan 을 다시 만든 뒤 full 생성. QA 게이트에서 멈춘다(exit=10).
pwsh -File .\run_pipeline.ps1 -Stage generate `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.measured.json `
  -Plan "E:\quality_fail_40k_plan2_v1.5\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_v1.5" -Detached

# 4) 사람이 fail_visual_qa.csv 를 채운 뒤 재개 → 최종 ZIP 공개
pwsh -File .\run_pipeline.ps1 -Stage resume `
  -RawRoot "E:\103.배터리 불량 이미지 데이터" -Config .\config.40k.measured.json `
  -Plan "E:\quality_fail_40k_plan2_v1.5\manifests\generation_plan.csv" `
  -Output "E:\quality_fail_40k_v1.5" -Detached

# 5) 원격(gdrive 등) 업로드
pwsh -File .\run_pipeline.ps1 -Stage upload `
  -Output "E:\quality_fail_40k_v1.5" -Remote "gdrive:quality_fail_40k_v1.5" -Detached
```

`generate`가 끝나면 visual-QA 게이트에서 `exit=10`으로 멈추며, 사람이 510장을 검토해
`manifests\fail_visual_qa.csv`의 `reviewer`와 `approved`를 채워야 한다. 스크립트는 이 사람
검토를 대신 수행하지 않는다. `upload`는 `rclone`이 설치되고 원격이 설정돼 있어야 하며,
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
- visual QA 승인율이 case별 95% 미만이면 릴리스 산출물을 만들지 않는다.

## 작업 기록

- `docs/작업노트_2026-07-24.md`: 첫 전량 실행에서 잡은 성능·안정성·품질 결함
- `docs/시각검수_결과_2026-07-27.md`: 1차 visual QA 전수 검수 결과와 미달 case 원인 분석
- `docs/작업노트_2026-07-28_visual_qa.md`: 회수본 재검사, 재생성 범위와 경로 산정

## 테스트

```powershell
$env:PYTHONPATH = (Resolve-Path ".\src").Path
python -m unittest discover -s tests -v
```
