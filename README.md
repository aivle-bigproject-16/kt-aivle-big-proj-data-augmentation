# quality-fail-augment 1.5

`CT RGB 품질 PASS FAIL 독립 데이터셋 생성 계획서 v1.5`의 실행 코드다.

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
- `augment.py`: CT/RGB failure case의 실제 영상 변환
- `geometry.py`: ROI, polygon parsing/clipping, affine 좌표 변환
- `generator.py`: 병렬 실행, 재시도, 파일 commit, manifest와 QA gate
- `cli.py`: `audit-raw`, `plan`, `generate`, `verify` 명령 진입점

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
않는다.

전체 생성:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.5"
```

전체 파일 생성·자동 검증 뒤에는 `manifests/fail_visual_qa.csv`가 생성되고 첫
실행이 `Visual QA approval pending`으로 중단된다. CT 8 case × 30장과 RGB
9 case × 30장, 총 510장을 검토해 `reviewer`와 `approved=true/false`를
입력한다. case별 승인율이 95% 이상일 때만 `--resume` 실행이 최종 ZIP과
완료 summary를 공개한다. 95% 미만이면 해당 case 파라미터를 조정하고 case
전체를 재생성해야 한다.

재개:

```powershell
quality-fail-augment generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality_fail_40k_plan_v1.5\manifests\generation_plan.csv" `
  --output "E:\quality_fail_40k_v1.5" `
  --resume
```

검증:

```powershell
quality-fail-augment verify --output "E:\quality_fail_40k_v1.5"
```

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

## 테스트

```powershell
$env:PYTHONPATH = (Resolve-Path ".\src").Path
python -m unittest discover -s tests -v
```
