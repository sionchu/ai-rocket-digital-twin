# ai-rocket-digital-twin (Clean-v0)

연구용 가상 실험실: 엔진/노즐 형상별 추진 성능 해석(Track A)과 6-DoF 자세 제어·VTVL 이착륙(Track B)을
버전이 붙은 `EnginePerformanceMap` + `VehicleConfig` + `ThrusterLayout` 인터페이스로만 연결한다.
실물 엔진 제작·시동·추진제 취급은 범위 밖이다.

검증 상태와 게이트 판정은 [`PROJECT_LEDGER.md`](PROJECT_LEDGER.md), 다음 액션은 [`HANDOFF.md`](HANDOFF.md),
결론은 [`docs/conclusion.md`](docs/conclusion.md).

## 빠른 시작

```bash
python -m pip install -e ".[test]"          # numpy 2.5.3, scipy 1.18.1, pyyaml 6.0.1
python -m rocket_dt validate-physics          # G0 단위 테스트 (pytest) -> runs/validate_physics/
python -m rocket_dt build-map --engine configs/engines/bell_B.yaml
python -m rocket_dt simulate --engine configs/engines/conical_A.yaml --scenario configs/scenarios/landing.yaml
python -m rocket_dt experiments --n-suite 30   # E0/E1/E2 + 착륙 스위트 -> runs/p0_experiments/
python -m rocket_dt cfd-case --engine configs/engines/conical_A.yaml --out runs/cfd --level 1
python scripts/run_cfd_batch.py --engines conical_A bell_B --levels 1 2 3   # OpenFOAM 필요
```

`train`, `finetune`, `evaluate`, `visualize`는 Isaac Lab/GPU가 필요한 후속 단계이며 아직 구현되지 않았다(NOT_RUN).

## 구조

| 경로 | 내용 |
|---|---|
| `src/rocket_dt/schemas.py` | SSOT 스키마, SI·좌표계 검사, 해시 |
| `src/rocket_dt/propulsion/` | 1D 이상 노즐, 윤곽 생성, 성능 맵(외삽 금지), 액추에이터, Propulsion Adapter |
| `src/rocket_dt/cfd/` | OpenFOAM wedge 케이스 생성기, polyMesh/필드 리더, 출구면 적분, QA, GCI, CFD 맵 |
| `src/rocket_dt/dynamics/` | 좌표계 규약, 질량·CoM·관성 모델, r×F, RK4 6-DoF 참조 적분기 |
| `src/rocket_dt/control/` | 제어 권한 검사, RCS 할당(bounded LS), cascaded PD 기준선 |
| `src/rocket_dt/isaac_env/` | Isaac wrench bridge(테스트됨), DirectRLEnv 골격(NOT_RUN) |
| `src/rocket_dt/evaluation/` | 시나리오 실행, 실험(E0/E1/E2/S3), 산출물(manifest/metrics) |
| `configs/` | 엔진 A/B/C, 기체 v0, 시나리오 |
| `docs/research/` | 단계 0 조사 보고서 5개 / `docs/sources.md` 통합 출처 |
| `docs/decisions/` | ADR-0001 … 0008 |
| `runs/` | 실행 산출물 (대용량 궤적·CFD 시간 디렉터리는 git 제외) |

## 규약 요약

SI 단위, world Z-up, body +Z = 기수, 쿼터니언 (w,x,y,z) body→world (ADR-0002).
성능 맵 밖 조회는 예외, 분리 위험 셀은 `valid=false` → 에피소드 `physics_valid=false`.
한 실행에 적분기 하나 (ADR-0003).
