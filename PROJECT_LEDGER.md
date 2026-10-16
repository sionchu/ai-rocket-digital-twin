# PROJECT_LEDGER

갱신: 2026-10-16. 표기: PASS / PARTIAL / FAIL / NOT_RUN / BLOCKED.
"시뮬레이션 판정"은 코드·테스트가 정의대로 동작했는지, "물리 타당성"은 그 결과를 실제 물리 근거(해석해, 실험, 독립 솔버)로 뒷받침했는지다. 프로젝트 전체 판정은 없다(전체 PASS 아님).

## 실행 환경 (결과마다 이 라벨을 붙인다)
| 라벨 | 내용 |
|---|---|
| `cloud-v1912` | Claude 클라우드 컨테이너, Ubuntu 24.04, 4 CPU, GPU 없음, OpenFOAM ESI v1912 (Ubuntu 아카이브), Python 3.13.16, numpy 2.5.3, scipy 1.18.1 |
| `wsl-v2606` | 사용자 PC (RTX 4070 Ti 12 GB, WSL Ubuntu 24.04, OpenFOAM v2606). **이 환경에서 실행한 결과는 아직 없음** |
| `win-isaac` | 사용자 PC `C:\dev\IsaacLab`. **실행한 결과 없음**. VRAM 12 GB는 Isaac Lab 2.3 권장(16 GB)보다 작다 |

이 저장소의 모든 수치 결과는 `cloud-v1912`에서 나왔다. v2606 결과와 섞지 않는다.

## 게이트
| 게이트 | 시뮬레이션 판정 | 물리 타당성 | 종합 | 증거 |
|---|---|---|---|---|
| G0 뉴턴 역학·추진제 소비·토크 | PASS (31/31 테스트) | PASS: 해석해와 비교(자유낙하, 정적 호버, τ=r×F, 각운동량·에너지 보존 1e-7, RK4 4차 수렴 비 16±25%, ε 면적비, 추력계수 독립식, E0 손계산) | **PASS** | `runs/validate_physics/`, `tests/` |
| G1 이론 노즐 vs CFD 수렴 | PASS: 3격자 모두 QA 통과, GCI 계산 | PARTIAL: 비점성 CFD vs 1D만 비교. 실험 데이터(Back–Massier–Gier, NPARC) 비교 NOT_RUN. 점성·분리 미모델. QA 규칙은 결과를 본 뒤 바꿨다(ADR-0006) | **PARTIAL** | `runs/cfd/cfd_summary.json`, `runs/cfd_experiments/summary.md` |
| G2 설계 ID → 맵 → Isaac → 3D 리플레이 | PARTIAL: 설계 ID → CFD 맵(해시) → Python 6-DoF 주입까지 동작 | Isaac 적용 BLOCKED(클라우드), CadQuery CAD NOT_RUN, 3D 리플레이 NOT_RUN | **PARTIAL** | `runs/cfd/maps/*.json`, `runs/cfd_experiments/` |
| G3 기준선 호버·자세·접촉 | PASS (Python 참조): 호버 위치 RMS 0.21 m, 착륙 30/30 × A/B/C | PARTIAL: 참값 상태, 센서 노이즈·지연 없음, 공력 없음, 접촉은 페널티 모델. Isaac 접촉 비교 NOT_RUN | **PARTIAL** | `runs/p0_experiments/summary.md` |
| G4 PPO 착륙 | BLOCKED (클라우드 GPU 없음) | — | **BLOCKED** (이관 경로: `win-isaac`) | ADR-0001, 03 보고서 |
| G5 파인튜닝 전·후 | BLOCKED | — | **BLOCKED** | — |
| G6 surrogate 오차·OOD | NOT_RUN (CFD 설계 2개뿐, ADR-0007은 8–12개 요구) | — | **NOT_RUN** | ADR-0007 |

## 검증 항목 (사용자 리뷰 2026-10-16 대응)
| 항목 | 판정 | 핵심 수치·근거 |
|---|---|---|
| E0 엔진 A/B/C 교체 | **PASS** (1D 맵 기준) | 같은 명령(스로틀 1.0, 짐벌 2°)에서 a_z: A 6.6234, B 6.6618(+0.58%), C 6.4738(−2.26%) m/s². α_y: A −0.47942, B −0.48104, C −0.47320 rad/s². 엔진 질량 47.19 / 46.79 / 48.70 kg. 모든 값이 손계산식 a_z = F cosδ/m − g0, α_y = τ_y/I_yy와 1e-9 이내로 일치(`test_e0_matches_hand_calculation`). 물리 타당성은 1D 맵 정확도에 묶임 |
| bell_C 노즐 분리 판정 | **PARTIAL** | 모델: Summerfield형 경험 기준 Pe/Pa < 0.4면 분리 위험(문헌 범위 0.25–0.4, 교재 원문 미확인 = INFERRED). bell_C는 Pe = 62.2 kPa × 스로틀이라 해면에서 스로틀 < 0.652면 위험. 호버 스로틀 약 0.64라 걸린다. 기준을 0.25로 바꾸면 경계가 0.41로 내려가 호버가 유효해진다. 즉 판정이 임계값에 민감하다. 분리를 실제로 푸는 CFD(점성, 외부 영역)는 NOT_RUN |
| PD 착륙 30/30 | **PARTIAL** | 30시드(1000–1029), Wilson 95% [0.89, 1.00]. 착지 위치 오차 P95 0.71 m, |vz| P95 0.83 m/s, 최대 착지 기울기 6.5°. 같은 시드 분포 안에서만 평가함. 분포 밖(OOD), 센서 노이즈, 공력은 미시험 |
| RCS 1개 고장 | **PASS** (수정 후) | 원인: 할당기가 τx·τy=0을 강하게 요구해 남은 +롤 추력기 하나를 쓰지 않았고, −롤만 작동해 롤 속도가 한쪽으로 누적됐다(최대 1.5 rad/s, 전도). 수정: pitch/yaw 부작용은 약하게만 벌점 주고 짐벌로 상쇄(feed-forward). 결과 rcs_01 고장 19/30 → 30/30 (고장 인지 시), 고장 미인지 시에도 30/30. A/B/C 모두 같음 |
| RCS 2개 고장(+롤 권한 상실) | **PARTIAL** | 롤 권한이 한쪽만 남으면 heading 유지를 포기하고 롤 감쇠만 하도록 전환: 고장 인지 시 30/30. **고장 미인지 시 19/30 (Wilson [0.46, 0.78])**. 고장 검출(FDIR)이 없어서다. 남은 취약점 |
| OpenFOAM CFD | **PARTIAL** (`cloud-v1912`) | wedge ±2.5°, 1셀 두께, front/back `wedge` 패치, 축 면 붕괴(크기 0), checkMesh "Mesh OK" 6/6. 수렴: 마지막 8기록 창에서 추력 리플 < 5e-4, 질량 불균형 ≤ 3e-4. 비점성·slip 벽 |
| 3단계 격자 GCI | **PASS** (수치) | 격자비 1.5. stream thrust GCI_fine: A 0.135% (p = 0.92), B 0.089% (p = 1.51). 점근비 1.000 / 1.001, 단조 수렴 |
| CFD vs 1D | 결과 | A: F −0.37%, ṁ −0.67% (목 유량계수 효과). B: F −1.88%. **1D에서 가정한 벨 λ = 0.985는 이 윤곽에서 CFD로 뒷받침되지 않는다.** 이 매개변수화의 bell_B는 conical_A보다 해면 추력이 1.3% 낮다 |
| Isaac Sim / PPO | **BLOCKED** (클라우드) | GPU 없음(`nvidia-smi` 없음). 사용자 PC로 이관 가능. 영구 차단 아님 |

## 고정 버전
ADR-0001 참고. 실제 사용: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, OpenFOAM v1912. 목표 고정: OpenFOAM v2606, Isaac Lab v2.3.2 + Isaac Sim 5.1.0, rsl-rl-lib 3.1.2, cea 3.3.4.

## 결정
ADR-0001 툴체인 · 0002 좌표계 · 0003 적분 소유자/backend/추력 적용 · 0004 P0 단순화 · 0005 착륙 판정(잠정) · 0006 CFD 설정 · 0007 surrogate 범위 · 0008 엔진 설계점
