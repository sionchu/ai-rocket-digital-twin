# ai-rocket-digital-twin (Clean-v0)

**Can a vehicle land on thrust control alone?** A VTVL (vertical take-off, vertical landing) digital twin
in NVIDIA Isaac Lab 3.0 / Isaac Sim 6.0, inspired by the cold-gas thruster package announced for the Tesla Roadster.
Simulation only.

▶ Video: [`runs/ppo_win-isaac/videos/story_v6_tipover.mp4`](runs/ppo_win-isaac/videos/story_v6_tipover.mp4)
(86 s: ground-start hop with ignition and flight-computer outputs, strong-wind landing, four thruster layouts,
PD vs RL, a real tip-over, results)

## Results, physics v1 (`win-isaac`, 60 landings per case)

Physics v1 adds aerodynamics with a wind profile, propellant burn that changes mass, CoM and inertia, and the
OpenFOAM CFD thrust map ([ADR-0011](docs/decisions/ADR-0011-physics-v1.md)). PD gains were re-tuned per scenario.

| Scenario | PD | PD re-tuned | RL trained on nominal | RL trained for the scenario |
|---|---|---|---|---|
| Nominal | 60 | 60 | 60 | – |
| Strong gusty wind | 41 | 44 | 27 | **56** |
| RCS thruster failure (unannounced) | **60** | 60 | 26 | 35 |
| Sensor noise + 20 ms latency | **60** | 60 | 60 | 6 (fine-tune failed) |
| Start 25–45 m from the pad | 37 | **60** | 13 | 30 |
| Ground-start hop: ignition, hover, land on a second pad | 60 | – | – | 60 |

Thruster layouts (RL, nominal): 3-engine 43, 4-engine 17, fixed engine + cold gas 41 (v0: 60 / 60 / 59).

- With the more realistic physics, the scenario-trained RL policy beats PD only in strong wind. A re-tuned PD wins
  or ties elsewhere; the RCS-failure, noise and layout fine-tunes fell short at the v0 iteration budget.
- In the hop, both land 60/60; RL uses about 19 % less propellant (67.4 vs 83.3 kg). Ignition delay (0.3 s) and
  thrust rise time (0.2 s) are assumptions ([ADR-0012](docs/decisions/ADR-0012-ground-start-hop.md)).
- Not modelled: landing-gear damage, plume–ground interaction, combustion. The flame is an NVIDIA Flow
  visualisation of the 1D nozzle output.

Details: [`runs/ppo_win-isaac/eval_v1/README.md`](runs/ppo_win-isaac/eval_v1/README.md),
[`PROJECT_LEDGER.md`](PROJECT_LEDGER.md).

## Results, physics v0 (fixed mass, no aerodynamics)

| Scenario | Fixed-gain PD | RL trained on nominal | RL trained for the scenario |
|---|---|---|---|
| Nominal | 60 | 57 | – |
| Strong gusty wind (≤ 1500 N) | 43 | 20 | 54 |
| RCS thruster failure (unannounced) | 60 | 27 | 57 |
| Sensor noise + 20 ms latency | 60 | 60 | 17 (fine-tune failed) |
| Start 25–45 m from the pad | 5 | 28 | 52 |

Thruster layouts (RL, nominal): gimballed engine + roll RCS 60, 3-engine differential throttle 60,
4-engine differential throttle 60, fixed engine + cold-gas attitude control 59.

- A well-tuned fixed-gain PD is as good or better inside its design envelope. RL wins only outside it,
  and only when that condition is in training.
- Limits: fixed vehicle mass (ADR-0009), 1D nozzle maps (CFD-checked for engines A and B), no aerodynamics,
  no hardware. PD was not re-tuned per scenario; layouts b–d have no PD baseline. The flame in the videos is
  an NVIDIA Flow visualisation of the 1D nozzle output, not a combustion simulation.

Details: [`docs/decisions/ADR-0010-scenarios-layouts.md`](docs/decisions/ADR-0010-scenarios-layouts.md).

---

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

Isaac Lab 학습·평가 (Windows, GPU 필요, `win-isaac`):

```bat
isaaclab.bat -p scripts\ppo_train.py --run_name r0_base --num_envs 4096 --max_iterations 150
isaaclab.bat -p scripts\ppo_train.py --run_name wind --scenario wind --resume_from runs\ppo_win-isaac\train\r3_guided\model_398.pt
isaaclab.bat -p scripts\ppo_train.py --run_name coldgas --layout coldgas --max_iterations 600
isaaclab.bat -p scripts\ppo_eval.py --checkpoint <model.pt> --scenario wind --layout gimbal_rcs [--video --enable_cameras]
```

전체 명령과 하이퍼파라미터는 [`HANDOFF.md`](HANDOFF.md), ADR-0009/0010.

## 구조

| 경로 | 내용 |
|---|---|
| `src/rocket_dt/schemas.py` | SSOT 스키마, SI·좌표계 검사, 해시 |
| `src/rocket_dt/propulsion/` | 1D 이상 노즐, 윤곽 생성, 성능 맵(외삽 금지), 액추에이터, Propulsion Adapter |
| `src/rocket_dt/cfd/` | OpenFOAM wedge 케이스 생성기, polyMesh/필드 리더, 출구면 적분, QA, GCI, CFD 맵 |
| `src/rocket_dt/dynamics/` | 좌표계 규약, 질량·CoM·관성 모델, r×F, RK4 6-DoF 참조 적분기 |
| `src/rocket_dt/control/` | 제어 권한 검사, RCS 할당(bounded LS), cascaded PD 기준선 |
| `src/rocket_dt/isaac_env/` | Isaac wrench bridge, VTVL DirectRLEnv, 시나리오·엔진 배치, PD 정책, Flow 화염 |
| `src/rocket_dt/evaluation/` | 시나리오 실행, 실험(E0/E1/E2/S3), 산출물(manifest/metrics) |
| `configs/` | 엔진 A/B/C, 기체 v0, 시나리오 |
| `docs/research/` | 단계 0 조사 보고서 5개 / `docs/sources.md` 통합 출처 |
| `docs/decisions/` | ADR-0001 … 0010 |
| `runs/` | 실행 산출물 (대용량 궤적·CFD 시간 디렉터리는 git 제외) |

## 규약 요약

SI 단위, world Z-up, body +Z = 기수, 쿼터니언 (w,x,y,z) body→world (ADR-0002).
성능 맵 밖 조회는 예외, 분리 위험 셀은 `valid=false` → 에피소드 `physics_valid=false`.
한 실행에 적분기 하나 (ADR-0003).
