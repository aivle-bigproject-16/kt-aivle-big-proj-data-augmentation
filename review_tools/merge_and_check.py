r"""검수자별 CSV를 병합하고 _visual_qa_gate 통과 여부를 미리 판정한다.

사용법:
    python merge_and_check.py reviews\*.csv
    python merge_and_check.py reviews\*.csv -o fail_visual_qa.filled.csv

generator.py::_visual_qa_gate 가 강제하는 조건을 그대로 재현해서,
resume 를 돌리기 전에 실패를 미리 잡아낸다.
"""
import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

QA_FIELDS = [
    "modality", "failure_case", "augmentation_subtype",
    "synthetic_id", "image_path", "reviewer", "approved", "reason",
]
TRUTHY = {"true", "yes", "1"}
VALID = {"true", "false", "yes", "no", "1", "0"}
MIN_RATE = 0.95


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != QA_FIELDS:
            raise SystemExit(
                f"[{path}] 헤더가 계약과 다릅니다.\n  기대: {QA_FIELDS}\n  실제: {reader.fieldnames}"
            )
        return list(reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="검수자별 CSV (glob 가능)")
    ap.add_argument("-b", "--base", default="fail_visual_qa.csv",
                    help="원본 CSV (ID 세트 기준)")
    ap.add_argument("-o", "--out", default="fail_visual_qa.filled.csv")
    args = ap.parse_args()

    here = Path(__file__).parent
    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = here / base_path
    base = read_rows(base_path)
    base_ids = [r["synthetic_id"] for r in base]
    base_set = set(base_ids)

    files = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        if not hits:
            print(f"경고: '{pattern}' 에 해당하는 파일 없음")
        files.extend(hits)
    if not files:
        raise SystemExit("입력 CSV가 없습니다.")

    print(f"입력 파일 {len(files)}개")

    # synthetic_id -> list[(reviewer, approved_bool, reason, source_file)]
    collected = defaultdict(list)
    for f in files:
        rows = read_rows(f)
        ids = {r["synthetic_id"] for r in rows}
        if ids != base_set:
            raise SystemExit(
                f"[{f}] synthetic_id 세트가 원본과 다릅니다 "
                f"(누락 {len(base_set - ids)}, 초과 {len(ids - base_set)}). "
                "행을 추가·삭제하지 마세요."
            )
        n = 0
        for r in rows:
            val = r["approved"].strip().casefold()
            if not val:
                continue
            if val not in VALID:
                raise SystemExit(
                    f"[{f}] {r['synthetic_id']}: approved='{r['approved']}' 는 허용값이 아닙니다 "
                    f"({sorted(VALID)})"
                )
            name = r["reviewer"].strip()
            if not name:
                raise SystemExit(f"[{f}] {r['synthetic_id']}: approved 는 있는데 reviewer 가 비었습니다.")
            collected[r["synthetic_id"]].append((name, val in TRUTHY, r["reason"].strip(), f))
            n += 1
        print(f"  {Path(f).name}: {n}행 판정")

    # 충돌 검사
    conflicts = []
    for sid, entries in collected.items():
        verdicts = {e[1] for e in entries}
        if len(verdicts) > 1:
            conflicts.append((sid, entries))
    if conflicts:
        print(f"\n판정 충돌 {len(conflicts)}건 — 사람이 합의해야 합니다:")
        for sid, entries in conflicts[:20]:
            detail = ", ".join(f"{e[0]}={'승인' if e[1] else '거부'}({Path(e[3]).name})" for e in entries)
            print(f"  {sid}: {detail}")
        raise SystemExit("충돌을 해소한 뒤 다시 실행하세요.")

    # 병합
    merged = []
    for r in base:
        out = dict(r)
        entries = collected.get(r["synthetic_id"])
        if entries:
            reviewers = sorted({e[0] for e in entries})
            reasons = [e[2] for e in entries if e[2]]
            out["reviewer"] = ";".join(reviewers)
            out["approved"] = "true" if entries[0][1] else "false"
            out["reason"] = " | ".join(dict.fromkeys(reasons))
        merged.append(out)

    judged = [r for r in merged if r["approved"].strip()]
    print(f"\n병합 결과: {len(judged)}/{len(merged)} 판정 완료 "
          f"({len(judged)/len(merged)*100:.1f}%)")

    # 케이스별 승인율 (게이트 재현)
    by_case = defaultdict(list)
    for r in merged:
        if r["approved"].strip():
            by_case[(r["modality"], r["failure_case"])].append(
                r["approved"].strip().casefold() in TRUTHY
            )

    total_cases = {(r["modality"], r["failure_case"]) for r in merged}
    print("\n케이스별 승인율:")
    failing, incomplete = [], []
    for key in sorted(total_cases):
        vals = by_case.get(key, [])
        expected = sum(1 for r in merged
                       if (r["modality"], r["failure_case"]) == key)
        if len(vals) < expected:
            incomplete.append((key, len(vals), expected))
            print(f"  {key[0]:4} {key[1]:34} {len(vals):3}/{expected} 판정  (미완료)")
            continue
        rate = sum(vals) / len(vals)
        mark = "OK " if rate >= MIN_RATE else "FAIL"
        print(f"  {key[0]:4} {key[1]:34} {rate*100:6.2f}%  {mark}")
        if rate < MIN_RATE:
            failing.append((key, rate))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = here / out_path
    # 파이프라인(generator.py)과 동일하게 BOM 포함으로 쓴다.
    # BOM 없이 쓰면 한글 Windows 의 Excel 이 CP949 로 읽어 글자가 깨진다.
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QA_FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"\n저장: {out_path}")

    print("\n" + "=" * 56)
    if incomplete:
        print(f"미완료 케이스 {len(incomplete)}개 — 아직 resume 불가")
        print("  (모든 행이 판정돼야 게이트가 통과합니다)")
        return 2
    if failing:
        print(f"승인율 95% 미만 케이스 {len(failing)}개:")
        for key, rate in failing:
            print(f"  {key[0]}/{key[1]}: {rate*100:.2f}%")
        print("  → 해당 case 파라미터를 조정하고 case 전체를 재생성해야 합니다.")
        return 1
    print("전 케이스 95% 이상 — resume 진행 가능")
    print(f"  이 파일을 output\\manifests\\fail_visual_qa.csv 로 덮어쓰고")
    print("  run_pipeline.ps1 -Stage resume 실행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
