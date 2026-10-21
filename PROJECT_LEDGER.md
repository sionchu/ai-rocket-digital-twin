# PROJECT_LEDGER

갱신: 2026-10-17. 표기: PASS / PARTIAL / FAIL / NOT_RUN / BLOCKED.
"시뮬레이션 판정"은 코드·테스트가 정의대로 동작했는지, "물리 타당성"은 그 결과를 실제 물리 근거(해석해, 실험, 독립 솔버)로 뒷받침했는지다. 프로젝트 전체 판정은 없다(전체 PASS 아님).

## 실행 환경 (결과마다 이 라벨을 붙인다)
| 라벨 | 내용 |
|---|---|
| `cloud-v1912` | Claude 클라우드 컨테이너, Ubuntu 24.04, 4 CPU, GPU 없음, OpenFOAM ESI v1912 (Ubuntu 아카이브), Python 3.13.16, numpy 2.5.3, scipy 1.18.1 |
| `wsl-v2606` | 사용자 PC, i5-13600KF (14코어/20스레드), WSL2 Ubuntu 24.04, OpenFOAM ESI v2606 (`/usr/lib/openfoam/openfoam2606`), Python 3.12.3, numpy 2.5.3, scipy 1.18.1. 저장소는 WSL 내부 파일시스템에서 실행 |
| `win-isaac` | 사용자 PC Windows 11, RTX 4070 Ti 12 GB, 드라이버 610.88. **Isaac Lab 3.0.0** (`v3.0.0-beta2.patch1-5-g418ca31b4`, `C:\dev\IsaacLab`) + **Isaac Sim 6.0.1-rc.7** (`C:\isaacsim`), PhysX, kit python 3.12.13, torch 2.10.0+cu128, warp 1.13.0, numpy 2.3.1. `isaaclab.bat -p`로 실행(conda 없음). 목표 고정(2.3.2 + 5.1.0)과 다르다 |

`runs/cfd/`, `runs/cfd_experiments/`, `runs/p0_experiments/`, `runs/validate_physics/`는 `cloud-v1912` 결과다. `wsl-v2606` 결과는 `runs/cfd_wsl-v2606/`, `win-isaac` 결과는 `runs/isaac_smoke_win-isaac/`에만 있다. 서로 섞지 않는다.

## 게이트
| 게이트 | 시뮬레이션 판정 | 물리 타당성 | 종합 | 증거 |
|---|---|---|---|---|
| G0 뉴턴 역학·추진제 소비·토크 | PASS (31/31 테스트) | PASS: 해석해와 비교(자유낙하, 정적 호버, τ=r×F, 각운동량·에너지 보존 1e-7, RK4 4차 수렴 비 16±25%, ε 면적비, 추력계수 독립식, E0 손계산) | **PASS** | `runs/validate_physics/`, `tests/` |
| G1 이론 노즐 vs CFD 수렴 | PASS: 3격자 모두 QA 통과, GCI 계산. v2606에서 1e-4 이내로 재현(`wsl-v2606`) | PARTIAL: 비점성 CFD vs 1D만 비교. 실험 데이터(Back–Massier–Gier, NPARC) 비교 NOT_RUN. 점성·분리 미모델. QA 규칙은 결과를 본 뒤 바꿨다(ADR-0006) | **PARTIAL** | `runs/cfd/cfd_summary.json`, `runs/cfd_experiments/summary.md` |
| G2 설계 ID → 맵 → Isaac → 3D 리플레이 | PARTIAL: 설계 ID → CFD 맵(해시) → Python 6-DoF 주입까지 동작. Isaac 단일 강체 힘·토크 적용 PASS. 1D 맵 → Isaac env(GPU 쌍선형 조회) 동작 (`win-isaac`) | CFD 맵으로 Isaac env 학습·평가 NOT_RUN, Isaac 평가 영상 있음, CadQuery CAD NOT_RUN, 3D 리플레이 NOT_RUN | **PARTIAL** | `runs/cfd/maps/*.json`, `runs/cfd_experiments/` |
| G3 기준선 호버·자세·접촉 | PASS (Python 참조): 호버 위치 RMS 0.21 m, 착륙 30/30 × A/B/C | PARTIAL: 참값 상태, 센서 노이즈·지연 없음, 공력 없음, 접촉은 페널티 모델. Isaac 접촉 비교 NOT_RUN | **PARTIAL** | `runs/p0_experiments/summary.md` |
| G4 PPO 착륙 | PASS (`win-isaac`): 같은 Isaac env에서 PD와 PPO를 비교했다. 혼합 엔진 PPO m1: A 56/60, B 57/60, C 57/60 (시드 1000–1029 + 2000–2029). Isaac PD: 180/180 | PARTIAL: PPO가 PD보다 좋지 않다(성공률·위치오차·연료). 보상에 PD 수평 법칙이 들어 있다(ADR-0009 4.1). bell_C는 모든 제어기가 물리 유효 0/60. 고정 질량(ADR-0009), ADR-0005 임계값 잠정 | **PARTIAL** | `runs/ppo_win-isaac/summary_mixed.md`, `summary.md` |
| G5 파인튜닝 전·후 | NOT_RUN (r1–r3은 보상 설정 미세조정이지 엔진·맵 파인튜닝이 아님) | — | **NOT_RUN** | — |
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
| OpenFOAM CFD 재현 (`wsl-v2606`) | **PASS** (재현) | 같은 케이스 생성기, 같은 6케이스. 차이 하나: v2606 blockMesh 기본값이 위상 병합이라 축의 겹친 꼭짓점이 병합되지 않아 면적 0 면 120–270개가 남고 checkMesh 2항목 실패. v1912 기본값과 같은 기하 병합(`blockMesh -merge-points`)을 v1912보다 새 버전에서만 쓰도록 `scripts/run_cfd_batch.py`에 넣었다(케이스 파일은 그대로). 이후 checkMesh "Mesh OK" 6/6, 축 패치 0면, QA PASS 6/6. 창 평균 추력·ṁ의 v1912 대비 상대차: conical_A L1/L2/L3 F −1.9e-6 / +3.8e-5 / −2.9e-5, ṁ −6.3e-6 / +2.9e-5 / −4.7e-5. bell_B F −2.5e-5 / −2.0e-5 / +3.3e-5, ṁ −2.4e-5 / −4.5e-5 / +6.0e-6. 모두 절대 상대차 < 1e-4, GCI_fine보다 한 자릿수 이상 작다. 6잡 병렬, 케이스 벽시간 합 11,973 s(최대 L3 4,128 s), 전체 1 h 9 min. `runs/cfd_wsl-v2606/compare_vs_cloud-v1912.json` |
| 3단계 격자 GCI (`wsl-v2606`) | **PASS** (수치) | stream thrust GCI_fine: A 0.068% (p = 1.41, 점근비 1.0004), B 0.117% (p = 1.31, 1.0007). ṁ GCI_fine: A 0.042% (p = 1.37), B 0.034% (p = 1.29). 단조 수렴. p는 v1912(A 0.92, B 1.51)와 다르다: 격자 간 차이가 1e-3 수준이라 1e-5 수준의 솔버 차이에도 p가 크게 흔들린다. 외삽 추력 차 A −0.056%, B +0.026%로 GCI 안. 맵 해시는 값이 미세하게 달라 다르다(A 197f75f8a2d29c82, B 861c24d92a84263f) |
| CFD vs 1D | 결과 | A: F −0.37%, ṁ −0.67% (목 유량계수 효과). B: F −1.88%. **1D에서 가정한 벨 λ = 0.985는 이 윤곽에서 CFD로 뒷받침되지 않는다.** 이 매개변수화의 bell_B는 conical_A보다 해면 추력이 1.3% 낮다 |
| Isaac 힘 스모크 (`win-isaac`) | **PASS** | 단일 강체 4 env, dt 5 ms × 10 스텝. MassModel(vehicle_v0 + conical_A, 추진제 가득) m 1207.186 kg, CoM z −0.2276 m, I diag(4016.0, 4016.0, 320.06)를 PhysX에서 다시 읽어 일치 확인. `batched_wrench` → `permanent_wrench_composer.set_forces_and_torques_index(positions=None, is_global=False)`. Reference6DoF(RK4, 같은 dt) 대비 상대오차: 선가속도 2.9e-7 / 9.5e-5 / 6.6e-5 / 2.6e-3, 각가속도 – (참값 0, Isaac 0) / 2.0e-7 / 2.0e-5 / 2.4e-3 (마지막은 ω0 = (0.2, −0.1, 0.8) rad/s 자이로 항, 적분기 차이). 기준 1% / 2%. 쿼터니언 순서: rocket_dt wxyz, Isaac Lab 3.x xyzw (ADR-0002의 '같다'는 3.x에서 틀림). 발견: `set_inertias_index`로 축대칭 관성(I_xx = I_yy)을 쓰면 PhysX `setCMassLocalPose pose is not valid`와 함께 동역학이 틀어진다. USD MassAPI로 리셋 전에 지정해서 우회. 변질량 env에서 다시 다뤄야 함. `runs/isaac_smoke_win-isaac/` |
| PPO 착륙 (`win-isaac`) | **PARTIAL** | 4096 env, rsl-rl-lib 5.0.1. r0(처음부터 150 iter) 0/30 → 셰이핑 3종 미세조정(각 200–250 iter). 수평 속도를 PD 법칙 v_ref = −0.43 e_h에 맞추는 'guided' 셰이핑(r3)만 위치 오차를 2 m 아래로 내렸다. 최종 r3 model_398: 29/30, 위치오차 중앙/P95 0.64/1.43 m (PD 0.36/0.71), 연료 73.8 kg (PD 78.8). 실패 3/60은 \|vz\| 2.03–2.04, vh 1.01로 한계 바로 밖. 체크포인트를 같은 시드로 골랐으므로 보류 시드로 다시 확인함. GPU 총 약 2.5 h. VRAM 약 2.1 GB/프로세스 |
| E3 엔진 교체 (`win-isaac`, 재학습 없음) | **FAIL** (강건성) | bell_B 20/30 ([0.49, 0.81]): 실패 10건 중 8건이 \|vz\| 2.00–2.13. bell_C 5/30, 물리 유효 0/30, 실패는 모두 위치오차 1.8–3.5 m. PD는 맵을 역산해서 A/B/C 모두 30/30. PPO 정책은 학습 엔진의 추력 맵에 묶여 있다 |
| E3 재시험: 혼합 엔진 학습 (`win-isaac`) | **PASS** (강건성, 1D 맵 기준) | 리셋마다 A/B/C 맵을 무작위로 골라 학습. 관측에 엔진 ID 없음. m1(r3에서 미세조정, 300 iter): A 56/60 [0.84, 0.97], B 57/60 [0.86, 0.98], C 57/60 [0.86, 0.98] (r3: 57 / 42 / 9). 착지 \|vz\| P95 0.94–1.26 m/s (r3 1.97–2.08). 대가: 연료 A 84.9 kg (r3 72.6, Isaac PD 70.3). 남은 실패는 위치오차 2 m 경계. m2(처음부터, 500 iter): 57 / 59 / 50, \|vz\| P95 ≈ 0.4로 매우 부드럽지만 연료 101–107 kg. bell_C 성공은 물리 유효 0/60이라 물리 증거가 아니다 |
| Isaac PD 기준선 (`win-isaac`) | **PASS** | `CascadedPD`를 바꾸지 않고 VtvlEnv 행동 인터페이스로 구동. A/B/C × 시드 1000–1029, 2000–2029 모두 30/30 (각 엔진 60/60 [0.94, 1.00]). \|vz\| P95 1.30 m/s, 위치오차 P95 0.68 m, 연료 70–73 kg. Python 참조(cloud-v1912)는 \|vz\| P95 0.83, 위치 P95 0.71: 시뮬레이터 차이(강체 접촉, 고정 질량, 접촉 직전 기록)가 착지 속도에 +0.47 m/s로 보인다 |
| Isaac 임무 영상 (`win-isaac`) | 결과 | 플룸(실제 추력·짐벌 반영), RCS 분사, 힘 화살표, 추적 카메라, 웹캐스트형 HUD(엔진 링, 원격측정, CFD Mach, Pe/Pa, 추력 맵 작동점, 한계 게이지, 판정). 화면의 모든 숫자는 Isaac 기록에서 나온다. `videos/pd_vs_ppo_m1_conical_A_s1004.mp4`, `videos/ppo_m1_bell_C_s1004_separation.mp4` |
| Isaac Flow 화염 영상 (`win-isaac`) | 결과 | NVIDIA Flow(omni.flowusd 110.0.3)가 헤드리스 RTX 캡처에 잡힌다. 화염은 1D 노즐 출력(추력→길이, Pe/Pa→퍼짐)의 시각 표현이며 연소 해석이 아니다. 하강 중인 차량에서 점·구 방출기는 화염이 노즐 위로 남아 박스 방출기로 바꿨다. 캡처 확인(화염색 픽셀이 있는 프레임 비율): conical_A 0.97–0.98, bell_C 0.69 (과팽창 Pe/Pa ≈ 0.42라 가장 가늘다). 엔진 근접 카메라, 고장 RCS 빨간 표시. `runs/ppo_win-isaac/videos/flow_*.mp4` + `.manifest.json` |
| 스토리 영상 (`win-isaac`) | 결과 | LinkedIn용 66 s, 6.4 MB, 영상 위주. 순서: 제목(Isaac 화면 위) → 학습 단계별 16 env 격자(같은 시드: iteration 0 0/16 전도, 50 0/16, 149 0/16 착지했으나 한계 초과, 398 16/16, 697 16/16) → bell_C로 엔진 교체(단일 엔진 정책 4/16, 혼합 엔진 정책 16/16) → 한 정책, 엔진 3종(노즐을 각 엔진 윤곽으로 4배 스케일 메시화, 근접 카메라) → PD vs PPO → 엔진별 착륙 수 → 노즐 형상 애니메이션. 끝난 env는 마지막 자세로 고정(시각만). `runs/ppo_win-isaac/videos/story_vtvl_isaac.mp4`, `scripts/render_story.py` |
| 시나리오별 학습 (`win-isaac`) | **PARTIAL** | 60회 착륙 수 PD / RL 기준 / RL 시나리오 학습: 강풍 43/20/54, RCS 고장 60/27/57, 센서 잡음 60/60/17(잡음 학습 실패), 원거리 접근 5/28/52. RL은 고정 이득 PD의 설계 범위 밖(강풍, 원거리)에서 앞선다. PD 재조정 비교 NOT_RUN. ADR-0010 |
| 엔진 배치 (`win-isaac`) | **PASS** (학습 가능) | 같은 기체에서 RL이 착륙 학습: 3엔진 차동 스로틀 60/60, 4엔진 60/60, 고정 주엔진 + 냉가스 자세 59/60. PD 기준선 없음(NOT_RUN). ADR-0010 |
| 분할 화면 영상 (`win-isaac`) | 결과 | 58.6 s, 8.8 MB: 학습 단계 → 시나리오 4분할 → 강풍·원거리 PD vs RL → 엔진 배치 4분할 → 결과 → 배치 도식. `runs/ppo_win-isaac/videos/story_v2_scenarios_layouts.mp4` |
| 물리 v1 재학습·평가 (`win-isaac`) | **PARTIAL** | ADR-0011 물리(공력, 가변 질량, CFD 맵)로 RL 8개 재학습, 60회 평가. PD 기본 / PD 재튜닝 / RL 기본 / RL 시나리오: nominal 60/60/60/-, 강풍 41/44/27/56, RCS 고장 60/60/26/35, 잡음 60/60/60/6, 원거리 37/60/13/30. 배치 RL 3엔진 43, 4엔진 17, 냉가스 41 (v0 60/60/59). v0의 "RL이 PD보다 낫다"는 강풍에만 남는다. 미세조정 실패: rcs_fail, noise, 배치 3종(iter를 v0와 같게 고정). 재튜닝 PD wind try1 38/60은 실패 기록. `runs/ppo_win-isaac/eval_v1/README.md` |
| Isaac 질량 갱신 | 결과 | `set_masses_index`는 읽기값만 바뀌고 GPU 동역학 불변(측정 a_z 6.76 vs 기대 7.94 m/s²). env v0는 질량 고정. 착지 때 7.1% 무거움(ADR-0009) |

## 고정 버전
ADR-0001 참고. 실제 사용: `cloud-v1912` Python 3.13.16, numpy 2.5.3, scipy 1.18.1, OpenFOAM v1912. `wsl-v2606` Python 3.12.3, OpenFOAM v2606. `win-isaac` Isaac Lab 3.0.0 + Isaac Sim 6.0.1-rc.7 (목표 고정과 다름). 목표 고정: OpenFOAM v2606, Isaac Lab v2.3.2 + Isaac Sim 5.1.0, rsl-rl-lib 3.1.2, cea 3.3.4.

## 결정
ADR-0001 툴체인 · 0002 좌표계 · 0003 적분 소유자/backend/추력 적용 · 0004 P0 단순화 · 0005 착륙 판정(잠정) · 0006 CFD 설정 · 0007 surrogate 범위 · 0008 엔진 설계점
