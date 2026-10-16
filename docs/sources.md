# 통합 출처 목록

확인 일자는 모두 2026-10-16. 상태: VERIFIED(원문 직접 확인), INFERRED, UNKNOWN. 트랙별 상세 근거는 `docs/research/0N_*.md`.

| ID | 출처 | URL | 상태 | 라이선스 | 쓰는 곳 |
|---|---|---|---|---|---|
| S1 | NASA Glenn CEA 소개 | https://www.nasa.gov/glenn/research/chemical-equilibrium-with-applications/ | NOT_RUN(이번에 재확인 안 함) | 공공 | 배경 |
| S2 | NASA CEA 3.3.4 문서 / PyPI `cea` | https://nasa.github.io/cea/ , https://pypi.org/project/cea/ | VERIFIED | Apache-2.0 | ADR-0001, P2+ 열화학 |
| S3 | OpenFOAM rhoCentralFoam (v2306/v2606 문서) | https://doc.openfoam.com/2606/tools/processing/solvers/rtm/compressible/rhoCentralFoam | VERIFIED | GPL | `cfd/` |
| S3b | openfoam.org v11: rhoCentralFoam → shockFluid | https://openfoam.org/?p=8212 | VERIFIED | GPLv3 | ADR-0001 |
| S4 | PhysicsNeMo darcy_fno 예제 | https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/darcy_fno/README.html | VERIFIED | Apache-2.0 | 후속 FNO |
| S5 | Isaac Lab RL 개념 (RSL-RL, 체크포인트) | https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html | VERIFIED | BSD-3 | 단계 4 |
| S6 | Isaac Lab DirectRLEnv 튜토리얼 | https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html | VERIFIED | BSD-3 | `isaac_env/` |
| S7 | Isaac Lab v2.3.2 rigid_object.py / wrench_composer.py | https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py | VERIFIED(소스) | BSD-3 | ADR-0003 |
| S7b | Isaac Lab v3.0.0-EA 릴리스 노트 | https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA | VERIFIED | BSD-3 | ADR-0001 |
| S7c | Isaac Sim 6.2 요구사항 (RT 코어 GPU 필수) | https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html | VERIFIED | NVIDIA | BLOCKED 근거 |
| S8 | RocketPy 1.13.0 (TVC/RCS 없음, 소스 grep) | https://pypi.org/project/rocketpy/ | VERIFIED | MIT | 교차검증 |
| S9 | Hall et al., NTRS 20160001840 (RCS phase-plane, jet select) | https://ntrs.nasa.gov/citations/20160001840 | VERIFIED(초록) | 공공 | RCS 할당 |
| S10 | Gaudet, Linares, Furfaro 2018, 6-DoF PPO landing | https://arxiv.org/abs/1810.08719 | VERIFIED | arXiv | 단계 4 |
| S11 | Jiang et al. 2024, RAJS | https://arxiv.org/abs/2407.15083 | VERIFIED | arXiv | 커리큘럼 |
| S12 | Isaac Lab PhysX↔Newton 정책 이전, DR 주의 | https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html | VERIFIED | BSD-3 | 단계 4 DR |
| S13 | NASA Glenn Rocket Thrust Equation | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-thrust-equation/ | VERIFIED | 공공 | `nozzle1d.py` |
| S14 | NASA Glenn Thrust Equations Summary | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-equations-summary | VERIFIED | 공공 | `nozzle1d.py` |
| S15 | Souza, Silva, Maximo 2022 (SBR/LARS) — **3-DoF** 종방향 모델 | https://sol.sbc.org.br/index.php/sbrlars/article/view/23169 | VERIFIED | 학회 | 설계안 인용 정정 필요 |
| R1 | NASA TN D-467 원추 노즐 λ=(1+cosα)/2 실험 | https://ntrs.nasa.gov/citations/19980227347 | VERIFIED | 공공 | conical_A 검증 |
| R2 | Back–Massier–Gier 1965 원추 노즐 벽압 실험 | https://ntrs.nasa.gov/api/citations/19650010083/downloads/19650010083.pdf | VERIFIED | 공공 | CFD 검증 후보 |
| R3 | NPARC CDV 축대칭 노즐 | https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html | VERIFIED | 공공 | CFD 검증 후보 |
| R4 | Celik et al. GCI 절차 | https://qa12.asme.org/wwwasmeorg/media/ResourceFiles/Shop/Journals/JFENumAccuracy.pdf | VERIFIED | 논문 | `cfd/post.py::gci` |
| R5 | Lee, Leok, McClamroch 2010 SO(3) 기하 제어 | (서지만, 이번에 확인 안 함) | UNKNOWN | — | `control/baseline.py` 자세 오차 정의 |
| R6 | Bodson 2002 제어 할당 비교 | https://my.eng.utah.edu/~bodson/code/Evaluation%20of%20optimization%20methods.pdf | VERIFIED | 논문 | RCS 할당 |
| R7 | Iafrate et al. 2025, 6-DoF PPO 1단 착륙 | https://doi.org/10.1016/j.actaastro.2024.11.028 | VERIFIED(기관 저장소) | Elsevier | 단계 4 |
| R8 | Wołczyk et al. 2024, fine-tuning과 망각 | https://arxiv.org/abs/2402.02868 | VERIFIED | arXiv | 단계 4 파인튜닝 (BC 정규화 우선) |
| R9 | Szmuk & Açıkmeşe 2018 SCvx 6-DoF 착륙 | https://arxiv.org/abs/1802.03827 | VERIFIED | arXiv | MPC/유도 기준선 |
| R10 | Ubuntu 24.04 `openfoam 1912.200626-2build3` | apt (archive.ubuntu.com) | VERIFIED(설치·실행) | GPL | 이번 CFD 실행 |
