# RGB test set 별도 생성

기존 전체 `generation_plan.csv`의 RGB main/test battery 분할을 사용해 RGB test
1,000장만 별도 폴더에 생성한다. 기본 구성은 증강 100장과 비증강 900장이다.

실행 전에 다음 항목이 RGB main과 겹치지 않는지 검사하며, 하나라도 겹치면 생성하지 않는다.

- 원본 `battery_id`와 발급된 battery ID
- 원본 이미지 경로
- 원본 이미지 SHA-256
- 디코딩된 이미지의 `pixel_hash`

```powershell
& "C:\Users\User\Documents\Codex\rgb-augmentation-venv\Scripts\python.exe" `
  -m quality_fail_augment.cli extract-rgb-test `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --config ".\config.40k.json" `
  --plan "E:\quality-fail-v2-plan\manifests\generation_plan.csv" `
  --output "E:\quality-fail-v2-rgb-test" `
  --total 1000 `
  --augmented 100 `
  --trust-plan
```

실패한 ID만 다시 생성할 때는 같은 명령에 `--resume`을 추가한다. 완료된 파일은
유지하고 누락된 ID만 같은 test battery 그룹의 reserve 원본으로 복구한다. RGB main의
원본 경로와 `pixel_hash`는 reserve 후보에서도 제외된다.

```powershell
  --resume `
  --trust-plan
```

`--trust-plan`은 전체 raw 재스캔만 생략한다. 선택된 1,000개 원본 이미지와 JSON의
SHA-256 검증은 그대로 수행한다. 출력 폴더에는 RGB test 이미지만 생성된다.

중복 검사 결과와 최종 수량은 다음 파일에 기록된다.

```text
E:\quality-fail-v2-rgb-test\manifests\rgb_test_selection_audit.json
```

출력 폴더가 이미 비어 있지 않으면 다른 새 폴더를 지정한다.
