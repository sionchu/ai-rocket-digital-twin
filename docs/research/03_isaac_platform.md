# Track 3 — Simulation & RL Platform

> 확인 일자: 2026-10-16. 소스 코드는 GitHub 태그(v2.3.0, v2.3.2, v3.0.0-EA)를 직접 내려받아 확인함. "VERIFIED"는 이번 조사에서 직접 열어 확인한 항목을 뜻함.

## 결론 (3–5 lines, Korean)
- **P0 고정 권고:** Isaac Lab **v2.3.2** + Isaac Sim **5.1.0**(pip `isaacsim[all,extscache]==5.1.0`) + Python **3.11** + **PhysX(GPU)** + rsl-rl-lib **3.1.2**. 2.x 계열의 최종 안정판이고 quadcopter 템플릿도 들어 있다. 차기 후보는 v3.0.0-EA(Isaac Sim 6.1, Python 3.12, uv)다. Isaac Sim 6.2.0(2026-10)은 "v3.0.0-GA"를 지원한다고 밝혔지만 GA 태그는 아직 없다.
- **추력 API:** `set_external_force_and_torque()`는 2.3.2에서 deprecated 경고를 띄우고 3.0에서 정식 deprecated다. 두 버전 모두에 있는 `asset.permanent_wrench_composer.set_forces_and_torques(...)`를 매 `_apply_action()`에서 호출하는 방식으로 표준화한다. 값은 reset 전까지 유지되고 `write_data_to_sim()`은 DirectRLEnv가 sim step마다 자동으로 호출한다.
- **주의 1:** `positions` 기준점이 버전마다 다르다. 2.3.x는 link frame 원점 기준이고 3.0은 CoM 기준 offset이다. 따라서 Propulsion Adapter가 ΣF와 Σ(r_i−r_CoM)×F_i를 body frame에서 직접 계산하고 `positions=None`으로 넘겨야 버전과 무관해진다.
- **주의 2:** 3.0에서 `Isaac-Quadcopter-Direct-v0`가 **삭제**되었다. 대신 `isaaclab_contrib`의 Multirotor + ThrusterCfg(할당행렬, 1차 추력 지연)가 가장 가까운 템플릿이다. Newton 백엔드는 ThrusterCfg를 지원하지 않으므로 PhysX를 유지한다.
- **GPU:** RT 코어가 있는 NVIDIA RTX GPU(VRAM 16GB 이상)가 필수다. A100/H100은 지원되지 않고 CPU만으로는 Isaac Sim을 실행할 수 없다. 3.0의 kit-less Newton 경로가 CPU에서 도는지는 확인하지 못했다.

## 근거 표
| 주장 | 출처 URL | 확인 일자 | 상태(VERIFIED/INFERRED/UNKNOWN) |
|---|---|---|---|
| 태그 목록: v2.3.0/2.3.1/2.3.2, v3.0.0-beta, -beta2, -beta2.patch1, v3.0.0-EA(최신)가 있고 v3.0.0 GA 태그는 없음. 브랜치는 main, develop, release/3.0.0 | https://github.com/isaac-sim/IsaacLab/releases (git ls-remote) | 2026-10-16 | VERIFIED |
| v3.0.0-EA(9월 16일): "This release is built for Isaac Sim 6.1"; "Start from a fresh Python 3.12 environment"; "`uv run` is now the recommended workflow"; `write_*_to_sim(data, env_ids)` 제거, `.data.*`가 더 이상 torch.Tensor가 아님 | https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA | 2026-10-16 | VERIFIED |
| v2.3.2(2월 2일): "This will be the final release from the current main branch". 2.3.2에서 main 브랜치의 rigid_object.py와 v2.3.2 태그 파일이 바이트 단위로 동일(cmp) | https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2 | 2026-10-16 | VERIFIED |
| v2.3.2 Migration: `set_external_force_and_torque()` deprecated. Permanent는 "persist across simulation steps until explicitly changed", Instantaneous는 "for a single simulation step, then automatically resets". `set_*`는 교체, `add_*`는 합성 | 같은 URL | 2026-10-16 | VERIFIED |
| Isaac Lab 2.3.x 설치 페이지: "We recommend using the latest Isaac Sim 5.1.0", Python 3.11, Ubuntu 22.04/Win11, RAM 32GB·VRAM 16GB 이상, Linux 드라이버 580.65.06 이상 | https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html | 2026-10-16 | VERIFIED |
| pip 경로: `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`, torch 2.7.0 cu128, GLIBC 2.35 이상, 이후 `./isaaclab.sh --install` | https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html | 2026-10-16 | VERIFIED |
| 3.0-EA pyproject: `requires-python >=3.12,<3.13`, `isaacsim[all,extscache]==6.1.0.0`, `rsl-rl-lib==5.4.1`. 설치 문서: "Isaac Sim 5.1 and older are not supported", 드라이버 580.95.05 이상 | v3.0.0-EA `pyproject.toml`, `docs/source/setup/installation/index.rst` | 2026-10-16 | VERIFIED |
| Isaac Sim 호환표: 6.1.0은 "Supports Isaac Lab v3.0.0-EA", 6.2.0(2026-10)은 "Supports Isaac Lab v3.0.0-GA" 및 Newton 1.6.1/MuJoCo-Warp 3.12.0 번들 | https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/compatibility.html | 2026-10-16 | VERIFIED |
| Isaac Sim 6.x 요구사항: 최소 RTX 4080·VRAM 16GB·RAM 32GB, 테스트 드라이버 595.58.03(Linux), "GPUs without RT Cores (A100, H100) are not supported". Ubuntu 22.04 지원은 deprecated | https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html | 2026-10-16 | VERIFIED |
| CPU만으로는 Isaac Sim(Kit/RTX)을 실행할 수 없음. `sim.device="cpu"`는 PhysX CPU 솔버 옵션일 뿐이며 이 경우에도 RTX GPU가 필요함 | 위 요구사항 + `simulation_cfg.py`(device 기본값 "cuda:0") | 2026-10-16 | INFERRED |
| 3.0 kit-less 경로(`uv run isaaclab train ... physics=newton_mjwarp`)는 Isaac Sim 없이 동작. CPU 단독 실행은 문서화되어 있지 않음 | v3.0.0-EA install index.rst | 2026-10-16 | UNKNOWN(CPU) |
| **2.3.2 시그니처:** `set_external_force_and_torque(self, forces, torques, positions=None, body_ids=None, env_ids=None, is_global=False)`. Docstring: "store the external force and torque into buffers which are then applied to the simulation at every step... positions... (in the local link frame of the bodies)". Note: "does not apply the external wrench... call write_data_to_sim right before the simulation step". 경고: "will be deprecated... use 'permanent_wrench_composer.set_forces_and_torques'" | https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py (L402-468) | 2026-10-16 | VERIFIED |
| 2.3.2 구현 이슈: 내부에서 `body_ids = self._ALL_BODY_INDICES_WP`로 덮어써 인자 `body_ids`가 무시됨. 단일 body인 RigidObject에서는 영향 없음 | 같은 파일 L457-458 | 2026-10-16 | VERIFIED(코드), 영향은 INFERRED |
| 2.3.0(구형): 자체 버퍼 `_external_force_b`를 쓰고, `is_global`을 바꾸면 전체 env에 마지막 프레임이 적용됨(caution). positions는 PhysX `apply_forces_and_torques_at_position`에 그대로 전달 | https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py | 2026-10-16 | VERIFIED |
| 2.3.2 composer docstring: permanent는 "persistent and are applied to the simulation at every step... thrust of a motor", instantaneous는 "only valid for the current simulation step... discarded" | v2.3.2 rigid_object.py L101-120 | 2026-10-16 | VERIFIED |
| 2.3.2 `WrenchComposer.add/set_forces_and_torques(forces, torques, positions, body_ids, env_ids, is_global=False)`: "The result is always in the link frame of the body". 커널 설명: is_global=False이면 positions는 link frame 기준 offset, True이면 월드 좌표 위치 | https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/utils/wrench_composer.py ; .../utils/warp/kernels.py L363-445 | 2026-10-16 | VERIFIED |
| `write_data_to_sim()`(2.3.2/3.0 공통): instantaneous+permanent를 합성한 뒤 `apply_forces_and_torques_at_position(position_data=None, is_global=False)`로 적용하고 매번 `instantaneous.reset()` 호출. `reset(env_ids)`는 두 composer 모두 초기화 | v2.3.2 rigid_object.py L134-168; v3.0.0-EA isaaclab_physx/assets/rigid_object/rigid_object.py L144-170 | 2026-10-16 | VERIFIED |
| 링크 원점 기준 토크(2.3.2)와 CoM 적용(position=None)이 섞여 CoM≠link 원점이면 모멘트 오차가 생김 | 위 코드 조합 | 2026-10-16 | INFERRED |
| **3.0-EA:** `set_external_force_and_torque`는 `DeprecationWarning`을 내고 내부적으로 "permanent_wrench_composer.reset followed by add_forces_and_torques"를 수행. `set_forces_and_torques`도 deprecated("4.5.33")되어 `set_forces_and_torques_index` / `add_forces_and_torques_index` / `*_mask`로 대체 | v3.0.0-EA isaaclab/assets/rigid_object/base_rigid_object.py L884-905; isaaclab/utils/wrench_composer.py L229-350, L638-657 | 2026-10-16 | VERIFIED |
| 3.0 positions 정의: "If `is_global` is False, these are offsets from the body's CoM expressed in the body frame. If None, forces are assumed to act at the body's CoM". `set_*_index`는 대상 env만 reset한 뒤 기록 | 같은 wrench_composer.py | 2026-10-16 | VERIFIED |
| DirectRLEnv.step: `_pre_physics_step` 1회 → decimation 루프마다 `_apply_action()` → `scene.write_data_to_sim()` → `sim.step()` → `scene.update()`. Newton이 decimation을 처리하는 경우(`handles_decimation`)에는 `_apply_action`이 1회만 호출됨 | v3.0.0-EA isaaclab/envs/direct_rl_env.py L400-436 | 2026-10-16 | VERIFIED |
| 백엔드: 3.0-beta는 "Full PhysX backend (default)"와 Newton을 제공. EA의 `SimulationCfg.physics` 기본값은 "None (uses PhysxCfg())". 2.3.x는 PhysX만 지원(Newton은 별도 beta 문서) | v3.0.0-beta 릴리스 노트; v3.0.0-EA sim/simulation_cfg.py L101-105 | 2026-10-16 | VERIFIED |
| S12 이전 가이드: `physics=isaacsim_physx` 또는 `physics=newton_mjwarp`로 선택. "Transfer works when both backends expose the same policy inputs and outputs". "Expect similar behavior, not identical trajectories". DR은 "plausible modeling uncertainty"를 넘지 않게. "Keep a separate deterministic nominal evaluation". "Zero gravity is not the deployment condition". 관절/바디 순서 차이 주의, Newton은 마찰계수 1개 | https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html | 2026-10-16 | VERIFIED |
| drone_arl 태스크 주석: "ThrusterCfg is implemented in Isaac Lab and has no Newton-native execution path" → `use_newton_actuators=False`, PhysxCfg 사용 | v3.0.0-EA isaaclab_tasks/contrib/drone_arl/.../navigation_env_cfg.py L341 | 2026-10-16 | VERIFIED |
| S6 DirectRLEnv 튜토리얼 URL이 유효함(메타 2.3.2): "Creating a Direct Workflow RL Environment" | https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html | 2026-10-16 | VERIFIED |
| rsl-rl-lib 최신 5.5.1(2026-09-09), BSD-3, Python 3.9 이상. Isaac Lab 2.3.2는 `rsl-rl-lib==3.1.2`, 3.0-EA는 `==5.4.1`로 고정 | https://pypi.org/project/rsl-rl-lib/ ; v2.3.2 source/isaaclab_rl/setup.py L49 | 2026-10-16 | VERIFIED |
| 2.3.2 CLI: `--resume`, `--load_run`, `--checkpoint`(scripts/reinforcement_learning/rsl_rl/cli_args.py). 3.0(isaaclab_rl 0.15.0): "**Breaking:** Unified checkpoint loading on `--checkpoint`. Removed RSL-RL `--load_run` and `--resume`". cfg 필드 `resume`, `load_run=".*"`, `load_checkpoint="model_.*.pt"`(정규식이면 알파벳순 최신)는 유지 | v2.3.2 cli_args.py; v3.0.0-EA isaaclab_rl/docs/CHANGELOG.rst; rsl_rl/rl_cfg.py L314-330 | 2026-10-16 | VERIFIED |
| rsl_rl v5.4.1 `OnPolicyRunner.load(path, load_cfg=None, strict=True, map_location=None)` → `actor/critic.load_state_dict(strict=strict)` | https://github.com/leggedrobotics/rsl_rl/blob/v5.4.1/rsl_rl/runners/on_policy_runner.py L145-161; algorithms/ppo.py L372-388 | 2026-10-16 | VERIFIED |
| obs/action 차원이 바뀌면 `strict=False`여도 PyTorch가 shape mismatch 에러를 냄(정규화기 버퍼 포함). 차원 변경 시 첫·마지막 레이어 수동 surgery 또는 재학습 필요 | PyTorch `load_state_dict` 동작 | 2026-10-16 | INFERRED |
| 라이선스: Isaac Lab은 BSD-3(isaaclab_mimic만 Apache-2.0). Isaac Sim 소스는 Apache-2.0이고, 실행 시 Omniverse 라이선스 동의가 필요 | IsaacLab README/LICENSE (EA); https://github.com/isaac-sim/IsaacSim | 2026-10-16 | VERIFIED |
| 3.0 알려진 제약: "Surface-gripper tasks require CPU simulation with physics=isaacsim_physx"; "Isaac Sim backends cannot be combined with OVPhysX and OVRTX" | v3.0.0-EA 릴리스 노트 | 2026-10-16 | VERIFIED |

## 후보 자산
| 이름 | URL | 버전 | 라이선스 | 용도 | 채택 권고 |
|---|---|---|---|---|---|
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | v2.3.2 (2026-02-02) | BSD-3 | P0 RL 프레임워크(DirectRLEnv) | **채택(P0 고정)** |
| Isaac Lab 3.0 | 같은 repo, tag v3.0.0-EA | 3.0.0-EA (09-16), GA 미출시 | BSD-3 | 멀티백엔드(PhysX/Newton), uv 워크플로 | P1 마이그레이션 후보. GA 이후 재평가 |
| Isaac Sim | https://docs.isaacsim.omniverse.nvidia.com/ (pip: pypi.nvidia.com `isaacsim`) | 5.1.0(Lab 2.3.2용) / 6.1.0(EA) / 6.2.0(GA용, 2026-10) | 소스 Apache-2.0 + Omniverse EULA | 물리(PhysX) 및 렌더링 | 5.1.0 채택(P0) |
| PhysX 백엔드 | (Isaac Sim 내장) | Isaac Sim 번들 | NVIDIA | RL 권위 적분기(single-integrator) | **채택(기본값)** |
| Newton / MuJoCo-Warp | 3.0 `isaaclab_newton` | Newton 1.5.0(6.1) / 1.6.1(6.2) | Apache-2.0(Newton, INFERRED) | 교차 백엔드 강건성 검증 | 보류(P2, ThrusterCfg 미지원) |
| Isaac-Quadcopter-Direct-v0 | https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py | v2.3.2 (3.0에서 삭제) | BSD-3 | 추력 적용 템플릿: `permanent_wrench_composer.set_forces_and_torques(body_ids=..., forces=thrust, torques=moment)` | **P0 rocket env 골격으로 채택** |
| isaaclab_contrib Multirotor + ThrusterCfg (drone_arl) | https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA/source/isaaclab_contrib | v2.3.2와 3.0-EA에 모두 있음 | BSD-3 | 할당행렬(6×N), 추력 1차 지연, rise/fall τ, 추력 slew 제한 | 참고/부분 채택(Propulsion Adapter 설계 참조) |
| RSL-RL | https://pypi.org/project/rsl-rl-lib/ | 3.1.2(Lab 2.3.2 고정), 최신 5.5.1 | BSD-3 | PPO 학습기 | 채택(Lab이 고정한 버전 사용) |
| Aerial Gym Simulator | https://github.com/ntnu-arl/aerial_gym_simulator | 2nd release (Isaac Gym 기반) | BSD-3 | 멀티로터 모터 모델 참고(Multirotor 원전) | 참고만 |
| OmniDrones | https://github.com/btx0424/OmniDrones | Isaac Sim 4.1.0, 유지보수 중단 | MIT | 드론 RL 참고 | 비채택 |
| Landing-Starships (community) | https://github.com/alxndrTL/Landing-Starships | Unity ML-Agents | 미표기 | TVC ±7° 및 RCS 보상 설계 아이디어 | 아이디어 참고만 |

**Propulsion Adapter 권고 계약(P0, INFERRED 설계):** 매 `_apply_action()`에서 per-thruster body-frame (F_i, r_i)를 받아 F=ΣF_i, τ=Σ(r_i−r_CoM)×F_i (+ 반토크)를 계산한다. 이어서 `robot.permanent_wrench_composer.set_forces_and_torques(forces=F[:,None,:], torques=τ[:,None,:], body_ids=[base_id], is_global=False)`를 호출하고 positions는 넘기지 않는다. 3.0 이전 시에는 `set_forces_and_torques_index`로 이름만 바꾸면 된다. episode reset에서는 `asset.reset(env_ids)`가 composer를 비운다.

## 미해결
1. Isaac Lab v3.0.0 **GA** 릴리스 일정과 실제 태그. Isaac Sim 6.2.0 문서는 GA를 언급하지만 2026-10-16 현재 태그가 없다.
2. Isaac Sim 5.1.0의 정확한 최소 GPU·드라이버 표. 6.x 요구사항 페이지만 확인했고, 5.1은 Isaac Lab 문서의 드라이버 580.65.06 이상만 확인했다.
3. 3.0 kit-less Newton(Warp) 경로의 CPU 단독 동작 여부. 문서화되어 있지 않아 GPU 없는 CI 스모크에 쓸 수 있는지 미확인이다.
4. 2.3.2에서 CoM≠link 원점일 때 positions 사용 시 생기는 모멘트 오차. 코드 해석상 추정이며 force-API 스모크 테스트로 검증해야 한다(positions 사용 vs 직접 r×F 계산 비교).
5. Newton `handles_decimation=True`일 때 instantaneous composer가 decimation 블록 전체에 적용되는지 여부. 그전까지 permanent composer만 사용한다.
6. 같은 강체에 여러 composer 호출(추력+공력+RCS)을 `add_*`로 합성할 때 2.3.2의 `body_ids` 무시 버그가 Articulation에도 있는지 확인이 필요하다.
7. Isaac Lab/Isaac Gym 기반 공개 **로켓/VTVL** 태스크는 찾지 못했다. GitHub 검색 API는 세션 정책으로 차단되었고 웹 검색 결과는 Unity/논문뿐이었다. 추가 탐색이 필요하다.
8. Newton 라이선스(Apache-2.0 추정)와 Isaac Sim pip wheel의 EULA 세부 조항은 미확인이다.
