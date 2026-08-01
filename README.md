# quality-fail-augment v2.0

CT/RGB 배터리 이미지를 사용해 PASS/FAIL 학습 데이터를 결정론적으로 생성하는
증강 파이프라인이다. v2.0은 내장 scan cache를 사용하므로 plan 단계에서 원본 폴더
전체 탐색, preflight, 전체 파일 stat 및 해시 계산을 수행하지 않는다.

## v2.0 데이터 구성

| 모달리티 | PASS | FAIL | 합계 | test | main |
|---|---:|---:|---:|---:|---:|
| CT | 18,000 | 2,000 | 20,000 | 1,000 | 19,000 |
| RGB | 9,500 | 10,500 | 20,000 | 1,000 | 19,000 |
| 합계 | 27,500 | 12,500 | 40,000 | 2,000 | 38,000 |

CT test는 PASS 900장, FAIL 100장이다. CT main은 PASS 17,100장,
FAIL 1,900장이다.

RGB test 1,000장은 모두 FAIL이며, RGB main은 PASS 9,500장,
FAIL 9,500장이다. RGB failure case의 test 쿼터는 전체 쿼터 비율에 맞춰
결정론적으로 배정된다.

동일한 원본 이미지는 최종 40,000장 안에서 한 번만 사용한다.

### CT FAIL 쿼터

| failure case | 전체 | test | main |
|---|---:|---:|---:|
| `ct_cell_alignment_failure` | 400 | 20 | 380 |
| `ct_acquisition_motion` | 400 | 20 | 380 |
| `ct_insufficient_projection_sampling` | 400 | 20 | 380 |
| `ct_low_signal_noise` | 400 | 20 | 380 |
| `ct_beam_hardening_metal_streak` | 400 | 20 | 380 |

### RGB FAIL 쿼터

| failure case | 전체 |
|---|---:|
| `rgb_trigger_timing_failure` | 1,208 |
| `rgb_uneven_lighting` | 1,418 |
| `rgb_reflection_glare` | 1,418 |
| `rgb_focus_failure` | 1,628 |
| `rgb_underexposure` | 1,417 |
| `rgb_overexposure` | 1,417 |
| `rgb_surface_dust` | 997 |
| `rgb_hair_contamination` | 997 |

## main/test 누수 방지

CT와 RGB 모두 `original_battery_id`를 하나의 그룹으로 취급한다. 같은 battery ID의
이미지는 전부 main 또는 test 한쪽에만 배정된다.

plan 생성 직전에 다음 교집합을 검사하며 하나라도 발견되면 실행을 중단한다.

- CT main/test `original_battery_id`
- RGB main/test `original_battery_id`

검사 결과는 `plan_metadata.json`의 아래 필드에도 기록된다.

- `ct_main_test_original_battery_overlap`
- `rgb_main_test_original_battery_overlap`

정상 plan에서는 두 값이 모두 `0`이다.

## 요구 환경

- Windows 10/11
- Python 3.10 이상
- PowerShell 7 (`pwsh`). Windows PowerShell 5.1은 BOM 없는 UTF-8 한글 경로를 깨뜨린다.
- 충분한 출력 디스크 공간
- 원본 데이터 폴더 하나. 그 폴더 바로 아래에 `3.개방데이터`가 있어야 한다.

## 설치

저장소 폴더에서 가상환경을 만들고 패키지를 설치한다. 전역 Python에 설치하면
numpy·Pillow 버전이 고정 핀으로 덮여 다른 작업이 깨지므로 반드시 가상환경을 쓴다.

```powershell
py -3.12 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m quality_fail_augment.cli --version
```

버전 출력은 `2.0`이어야 한다.

## 경로 설정 (`.env`)

원본 데이터와 출력 폴더 위치는 머신마다 다르다. 매번 CLI 인자로 적는 대신
저장소 루트의 `.env`에 한 번 적어둔다. `.env`는 git에 올라가지 않는다.

```powershell
Copy-Item .env.example .env
```

복사한 `.env`를 자기 경로로 고친다.

```ini
QFA_RAW_ROOT=C:\Users\rudtn\Downloads\빅프로젝트_데이터\데이터 전처리\103.배터리 불량 이미지 데이터
QFA_CONFIG=./config.40k.json
QFA_PLAN_DIR=C:\...\quality-fail-v2-plan
QFA_OUTPUT_DIR=C:\...\quality-fail-v2-output
QFA_SMOKE_DIR=C:\...\quality-fail-v2-smoke
QFA_AUDIT_DIR=C:\...\quality-fail-v2-audit
```

| 변수 | 채우는 인자 |
|---|---|
| `QFA_RAW_ROOT` | `--raw-root` |
| `QFA_CONFIG` | `--config` |
| `QFA_PLAN_DIR` | `plan`의 `--output` |
| `QFA_SMOKE_DIR` | smoke 생성의 `--output` |
| `QFA_OUTPUT_DIR` | `generate`·`verify`의 `--output` |
| `QFA_AUDIT_DIR` | `audit-raw`의 `--output` |
| `QFA_PLAN_CSV` | `--plan`. 생략하면 `QFA_PLAN_DIR\manifests\generation_plan.csv` |

값에는 이스케이프 처리를 하지 않으므로 Windows 경로의 백슬래시를 그대로 적으면 된다.
우선순위는 CLI 인자, 셸 환경변수, `.env` 순이다. 다른 `.env`를 쓰려면
`--env-file` 인자나 `QFA_ENV_FILE` 환경변수를 준다.

## 내장 scan cache

v2.0 저장소에는 다음 캐시가 포함된다.

```text
cache/scan_cache.csv
```

캐시 정보:

- 전체 pair: 276,170개
- 유효 CT: 75,952개
- 유효 RGB: 200,218개
- SHA-256:
  `A755968A50EFD3E0F440822B65F09A35B5AE9EBC62A22A461459B2A250AB69DF`

`config.40k.json`의 `use_embedded_scan_cache`가 `true`이므로 `plan` 실행 시
`--reuse-scan`을 지정할 필요가 없다. 캐시가 없으면 전체 스캔으로 조용히
전환하지 않고 오류로 중단한다.

내장 캐시 모드는 다음 작업을 생략한다.

- `os.walk` 기반 원본 폴더 전체 탐색
- preflight
- 전체 원본의 stat 및 SHA-256 재계산
- 전체 이미지 디코딩 및 JSON 재검증

`generate --trust-plan`은 전체 raw 재스캔을 생략하지만 plan에서 실제로 선택한
원본 파일은 SHA-256으로 다시 확인한다.

> 내장 캐시는 현재 원본 데이터 스냅샷 전용이다. 원본 파일을 추가·삭제·수정했다면
> 캐시를 그대로 신뢰하지 말고 `use_embedded_scan_cache`를 `false`로 바꾼 후
> 새로운 scan cache와 plan을 생성해야 한다.

## 실행 방법

아래 명령은 저장소 루트에서 실행하며, 경로는 모두 `.env`에서 온다.
`.env` 값 대신 다른 경로를 쓰고 싶으면 해당 인자를 직접 주면 된다.

### 1. plan 생성

plan 출력 폴더는 새 폴더이거나 비어 있어야 한다.

```powershell
.\.venv\Scripts\python.exe -m quality_fail_augment.cli plan
```

주요 출력:

```text
<QFA_PLAN_DIR>\
├─ plan_metadata.json
├─ logs\
└─ manifests\
   ├─ generation_plan.csv
   ├─ reserve_sources.csv
   ├─ extraction_audit.csv
   └─ scan_cache.csv
```

`plan_metadata.json`에서 다음 값이 모두 `0`인지 확인한다.

```json
{
  "ct_main_test_original_battery_overlap": 0,
  "rgb_main_test_original_battery_overlap": 0
}
```

### 2. 소량 smoke 생성

본 생성 전에 모달리티별 100장을 만들어 환경과 메모리를 확인한다.
smoke는 출력 폴더가 다르므로 `--output`만 직접 준다.

```powershell
.\.venv\Scripts\python.exe -m quality_fail_augment.cli generate `
  --output $env:QFA_SMOKE_DIR `
  --limit-per-modality 100 `
  --trust-plan
```

### 3. 전체 40,000장 생성

출력 폴더는 존재하지 않거나 비어 있어야 한다.

```powershell
.\.venv\Scripts\python.exe -m quality_fail_augment.cli generate --trust-plan
```

### 4. 중단된 생성 재개

```powershell
.\.venv\Scripts\python.exe -m quality_fail_augment.cli generate --resume --trust-plan
```

이미 manifest에 정상 커밋된 `synthetic_id`는 건너뛰고 나머지만 생성한다.

### 5. 결과 검증

```powershell
.\.venv\Scripts\python.exe -m quality_fail_augment.cli verify
```

### run_pipeline.ps1

`run_pipeline.ps1`도 같은 `.env`를 읽는다. 로그·상태 기록과 분리 실행(`-Detached`)이
필요하면 이쪽을 쓴다.

```powershell
pwsh -File .\run_pipeline.ps1 -Stage plan -Detached
pwsh -File .\run_pipeline.ps1 -Stage generate -TrustPlan -Detached
pwsh -File .\run_pipeline.ps1 -Stage verify
```

## 주요 산출물

```text
<QFA_OUTPUT_DIR>\
├─ CT\
│  ├─ main\
│  └─ test\
├─ RGB\
│  ├─ main\
│  └─ test\
├─ manifests\
│  ├─ dataset_manifest.csv
│  ├─ generation_plan.csv
│  ├─ generation_errors.csv
│  └─ lineage_private.csv
├─ generation_summary.json
└─ augmentation_json_4k_v2.0.zip
```

각 PASS 샘플은 이미지와 label JSON을 갖는다. 각 FAIL 샘플은 이미지, label JSON,
failure case 하나를 기록한 augmentation JSON을 갖는다.

## 안전 및 결정론

- RNG는 고정 seed와 NumPy `PCG64`를 사용한다.
- 하나의 FAIL 이미지에는 failure case 하나만 적용한다.
- 출력 battery/image ID와 `synthetic_id`는 plan에서 고정한다.
- plan과 config hash가 다르면 생성을 중단한다.
- `--trust-plan`에서도 선택된 원본 이미지와 JSON의 SHA-256을 확인한다.
- 생성 후 이미지, label, polygon, 해시, 수량 및 case 쿼터를 자동 검증한다.
- 생성 실패 시 같은 모달리티의 reserve source로 교체를 시도한다.

## 테스트

```powershell
$env:PYTHONPATH = (Resolve-Path ".\src").Path

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

현재 전체 테스트 48개가 통과한다.

## Git LFS와 v2.0 브랜치 push

`cache/scan_cache.csv`는 약 160MB이므로 Git LFS가 필요하다.

```powershell
git lfs install
git lfs track "cache/scan_cache.csv"
git check-attr filter -- "cache/scan_cache.csv"

git add --all
git status
git lfs ls-files
git commit -m "feat: add v2.0 CT RGB augmentation pipeline with embedded scan cache"
git push -u origin v2.0
```

`git check-attr` 결과는 다음과 같아야 한다.

```text
cache/scan_cache.csv: filter: lfs
```

저장소:

```text
https://github.com/aivle-bigproject-16/kt-aivle-big-proj-data-augmentation.git
```

## 코드 구조

- `src/quality_fail_augment/planner.py`: 내장 캐시 복원, 원본 선택, ID 그룹
  분리, 쿼터 배정, plan 생성
- `src/quality_fail_augment/augment.py`: CT/RGB failure case 영상 변환과 품질 gate
- `src/quality_fail_augment/preprocessing/stages.py`: 전처리와 좌표 변환
- `src/quality_fail_augment/generator.py`: 병렬 생성, 재개, reserve 교체, 검증과 ZIP 생성
- `src/quality_fail_augment/cli.py`: `audit-raw`, `plan`, `generate`, `verify` 진입점
- `src/quality_fail_augment/settings.py`: `.env` 로딩과 경로 기본값 해석
- `config.40k.json`: v2.0 수량, case 쿼터, seed 및 출력 설정
- `.env.example`: 경로 설정 템플릿. `.env`로 복사해 쓴다
- `cache/scan_cache.csv`: 내장 v3 scan cache
