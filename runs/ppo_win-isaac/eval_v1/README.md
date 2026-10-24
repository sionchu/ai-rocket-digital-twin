# Physics v1 재학습·평가 (`win-isaac`, ADR-0011)

- 환경: DESKTOP-4F67J8P, RTX 4070 Ti 12 GB, Isaac Lab 3.0.0 / Isaac Sim 6.0.1-rc.7, 2026-10-22.
- 코드: `pc/physics-v1` (cloud/physics-v1 + aero dtype 수정 308f549 + wind 재튜닝 try2 3478c13). 실행 스크립트 `run_v1.ps1`, 집계 `scripts/summarize_eval_v1.py`.
- 물리: `--physics v1` (공력, 가변 질량 보상 렌치, conical_A CFD 맵). v0 숫자는 ADR-0010/README 그대로이고 물리 버전이 달라 같은 열에 놓았을 뿐 직접 비교는 아니다.
- 평가: 시드 1000–1029 + 2000–2029, 시나리오마다 60회, 결정적 정책. 원본 `<scenario>/<controller>/s{1000,2000}/eval_*.json`, 집계 `summary_v1.{json,md}`.

## 준비 (VERIFIED)
- pytest 첫 실행 46/47. `test_aero_varmass_mass_torch_match_numpy` 실패: `aero.py`의 축방향 Cd 계산에서 torch가 `float × bool` 텐서를 float32로 올려 float64 비교가 4e-8 상대 오차로 어긋남. `(u_ax > 0) + 0.0 * u_ax`로 dtype을 유지하도록 고친 뒤 47/47 통과. env는 float32라 학습 수치 영향은 없다고 판단(INFERRED).

## 스모크 (VERIFIED, n=8, `smoke/`)
- PD 8/8 착지. r3 model_398(v0 학습) 0/8, 전부 착지했지만 착지 수직속도 p95 5.4 m/s로 한계 초과.
- NaN 없음. 질량 1207.07 kg → 1113–1141 kg, 증가는 에피소드 리셋 8번뿐(1207.19 kg 복귀).
- 공력 `last_F_aero_b`: nominal 바람(기준고도 2.6–6.1 m/s) 중앙값 51–281 N, 최대 110–602 N. wind 시나리오(14–18 m/s) 중앙값 880–1560 N.

## 결과 (VERIFIED, 60회 성공 수)
| 시나리오 | PD 기본 v0 → v1 | PD 재튜닝 v1 | RL 기본 v0(r3) → v1(v1_nominal) | RL 시나리오 v0 → v1 |
|---|---|---|---|---|
| nominal | 60 → 60 | 60 | 57 → 60 | - |
| wind | 43 → 41 | 44 (try2, `wind.json`) | 20 → 27 | 54 → 56 |
| rcs_fail | 60 → 60 | 60 | 27 → 26 | 57 → 35 |
| noise | 60 → 60 | 60 | 60 → 60 | 17 → 6 |
| lateral | 5 → 37 | 60 | 28 → 13 | 52 → 30 |

| 배치 (RL, nominal) | v0 | v1 |
|---|---|---|
| tri_diff | 60 | 43 |
| quad_diff | 60 | 17 |
| coldgas | 59 | 41 |

- wind 재튜닝 PD try1(`wind_try1_8seeds.json`): 38/60. 실패 기록으로 보관(`wind/pd_tuned_try1/`). Isaac 결과 41/38/44가 클라우드 참조 적분기(cloud-v1912)의 41/38/44와 같다.
- 실패 사유별 집계는 `summary_v1.md`.

## 학습 (VERIFIED, 마지막 iter의 학습 중 성공률, 확률적 정책)
| 런 | 시작 | iter | 학습 성공률 | 시간 | 평가 판정 |
|---|---|---|---|---|---|
| v1_nominal | r3 model_398 | 300 | 0.997 | 10 분 | 목표 달성 (60/60) |
| v1_wind | v1_nominal model_697 | 300 | 0.959 | 10 분 | 목표 달성 (56/60) |
| v1_rcs_fail | 〃 | 300 | 0.781 | 10 분 | 실패: v0 57 → 35, PD 60에 못 미침 |
| v1_noise | 〃 | 300 | 0.282 | 10 분 | 실패: 6/60 (v0 noise 미세조정 실패 17/60과 같은 양상) |
| v1_lateral | 〃 | 400 | 0.836 | 13 분 | 부분: 30/60, 재튜닝 PD 60에 못 미침 |
| v1_tri_diff | 처음부터 | 600 | 0.892 | 20 분 | 실패: 43/60 (v0 60) |
| v1_quad_diff | 처음부터 | 600 | 0.880 | 20 분 | 실패: 17/60 (v0 60) |
| v1_coldgas | 처음부터 | 600 | 0.830 | 20 분 | 실패: 41/60 (v0 59) |

## 해석 (INFERRED)
- v0 결론 "시나리오 학습 RL이 PD보다 낫다"는 v1에서 wind 하나만 남는다. lateral은 기본 PD도 v1에서 37/60으로 올랐고 재튜닝 PD는 60/60이다. rcs_fail과 noise는 PD가 60/60으로 앞선다.
- 학습 iter를 v0와 같게 고정했기 때문에 가변 질량·공력으로 어려워진 과제(배치 b–d, rcs_fail, lateral)는 수렴 전에 멈췄을 가능성이 있다. 학습 성공률(0.83–0.89)과 결정적 평가 성공률(28–72 %)의 차이가 커서, 특히 quad_diff는 결정적 정책의 착지 판정 경계 문제일 수 있다. 원인은 확인하지 않았다(UNKNOWN).

## 하지 않은 것
- iter 증가, 보상 변경, 재시도 학습: NOT_RUN (지시 범위 밖).
- 렌더링: NOT_RUN (새 기체 에셋 준비 후 별도 지시).
- 체크포인트는 기존 런과 같이 `train/v1_*/`에 커밋했다. 학습 로그는 `../v1_*.log`.
- 학습 단계의 `status.jsonl` 기록은 감시 스크립트의 파일 잠금 때문에 wind–tri_diff 5건이 빠졌다. 위 학습 표는 각 런의 로그(`../v1_*.log`)와 `manifest.json`에서 다시 뽑았다.
