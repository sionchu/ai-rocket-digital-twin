# Track 5 — Papers, Prior Code, Datasets, Assets

> 조사일: 2026-10-16. 논문에 나온 성능 수치는 참고용이며, 프로젝트 목표치가 아닙니다.
> 출처 등급: [P]=peer-reviewed/arXiv, [O]=official, [OSS]=오픈소스, [C]=community/thesis

## 결론
- S10(Gaudet·Linares·Furfaro 2018, 6-DoF PPO 착륙)과 S4(PhysicsNeMo Darcy FNO 예제)는 실제로 존재하고 설명된 내용과 맞음. S15의 정식 출처는 SBC OpenLib(SBR/LARS 2022, pp.55-60)이며 **3-DoF 종방향 모델**임. 설계 문서에서 "6-DoF TVC"로 인용했다면 바로잡아야 함. DOI와 arXiv는 찾지 못함.
- 6-DoF PPO 착륙의 최근 근거는 Iafrate et al. 2025(Acta Astronautica, DOI 10.1016/j.actaastro.2024.11.028)가 가장 강함. "엔진 B로 fine-tune하면서 A 유지" 기준선은 Wołczyk et al. 2024(arXiv 2402.02868)에 따라 **BC 정규화 또는 kickstarting 우선, EWC는 보조**로 둘 것을 권고.
- 노즐 surrogate 공개 데이터로는 HF `taarg/MultiFidelity_Nozzle_JetFlows`(Apache-2.0, RANS 92건/LES 18건)와 `JinglaiZheng/Laval_nozzle`(CC-BY-4.0, 설명 없음)를 찾음. 둘 다 로켓 노즐 성능(Isp, 추력)을 직접 담고 있지 않아, 1차 GP/MLP 기준선은 자체 생성 데이터(준1D 등엔트로피 해석해 또는 저비용 CFD)가 현실적임.
- 공개 착륙 RL 코드는 대부분 **2D**임(EmbersArc MIT, jiupinjia CC BY-NC-SA). 6-DoF는 Tuxliri/RL_Rocket_6DOF가 있으나 **라이선스가 없어** 재사용 불가하고 참고만 가능. 3D 자산은 NASA 3D Resources GLB(저작권 없음, 단 NASA 휘장 제거 필요)와 자체 USD 프리미티브 조합을 권고. Isaac Sim 에셋은 NVIDIA 라이선스상 재배포가 제한됨.

## 근거 표
| 주장 | 출처 URL | 확인 일자 | 상태(VERIFIED/INFERRED/UNKNOWN) |
|---|---|---|---|
| [S10] "Deep RL for Six Degree-of-Freedom Planetary Powered Descent and Landing", Gaudet, Linares, Furfaro, arXiv 2018-10-20. 6-DoF, PPO, 엔진별 추력 명령, terminal/shaping reward에 서로 다른 할인율 적용, 센서 노이즈와 파라미터 불확실성에 강건 [P] | https://arxiv.org/abs/1810.08719 | 2026-10-16 | VERIFIED |
| [S15] 정식 제목: "Thrust Vectored Rocket Landing Integrated Guidance and Control with Proximal Policy Optimization", G. A. Souza, O. M. Silva, M. R. O. A. Maximo, SBR/LARS 2022 Anais pp.55-60 (2022-10-18) [P] | https://sol.sbc.org.br/index.php/sbrlars/article/view/23169 | 2026-10-16 | VERIFIED |
| S15 내용: **3-DoF 종방향** 모델(질량 변화, 착륙 다리, 확률적 바람). PPO 사용. 행동은 주추력, 노즐각, 측면추력 3개. 시뮬레이션만 수행. 초록에 정량 결과 없음 | 위와 동일 | 2026-10-16 | VERIFIED |
| S15 DOI/arXiv 없음(SBC 페이지에 DOI 표기 없음) | 위와 동일 | 2026-10-16 | UNKNOWN |
| [S4] PhysicsNeMo darcy_fno: 2D Darcy, FNO, 학습 중 on-the-fly 데이터 생성, 단일 GPU, `train_fno_darcy.py`. 지표와 버전 표기 없음(latest docs) [O] | https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/darcy_fno/README.html | 2026-10-16 | VERIFIED |
| PhysicsNeMo 라이선스 Apache-2.0, 약 3.4k stars [OSS] | https://github.com/NVIDIA/physicsnemo | 2026-10-16 | VERIFIED |
| Iafrate, Brandonisio, Hinz, Lavagna, "Propulsive landing of launchers' first stages with Deep RL", Acta Astronautica 227 (2025) 40-56. 3-DoF와 6-DoF 시뮬레이터, PPO, 클라우드 학습 파이프라인 [P] | https://re.public.polimi.it/handle/11311/1279157 | 2026-10-16 | VERIFIED |
| Carradori, "6-DOF Atmospheric Rocket Landing Guidance using Meta-RL", TU Delft MSc 2024. GTrXL이 LSTM보다 우수, 대기 중 6-DoF, 핀과 짐벌 사용 [C:thesis] | https://repository.tudelft.nl/record/uuid:bf2a598c-9694-40cd-8dec-03b73d539b54 | 2026-10-16 | VERIFIED |
| Leiva-Vélez et al., "Adversarial RL for Robust Planetary Landing", arXiv 2610.20252 (2026-10-14). 3-DoF, 적대적 외란 에이전트, 예비 결과만 있음 [P] | https://www.alphaxiv.org/abs/2610.20252 | 2026-10-16 | VERIFIED |
| Santos et al., "QuadRocket" arXiv 2607.02474. TVC 저비용 실기 테스트베드. **RL이 아닌** 적응 backstepping 제어, 실비행 검증 [P] | https://www.alphaxiv.org/abs/2607.02474 | 2026-10-16 | VERIFIED |
| [S11] RAJS(Jiang et al. 2024)는 LandSpace 제공 Simulink 3D 모델을 사용하며, 논문에 코드 링크 없음 | https://arxiv.org/html/2407.15083v1 | 2026-10-16 | VERIFIED |
| Wołczyk et al., "Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem", arXiv 2402.02868 v3(2024-07). EWC, BC, Kickstarting, Episodic Memory 비교. 로봇 과제에서는 BC가 최선, 그다음 EM과 EWC. NetHack에서는 KS가 최선. EWC는 다른 방법에 일관되게 밀림 [P] | https://arxiv.org/html/2402.02868v3 | 2026-10-16 | VERIFIED |
| 2402.02868의 학회 게재(ICML 2024) | 페이지에 표기 없음 | 2026-10-16 | INFERRED |
| Kirkpatrick et al., EWC, arXiv 1612.00796 (PNAS DOI) [P] | https://arxiv.org/abs/1612.00796 | 2026-10-16 | VERIFIED |
| Wołczyk et al., "Continual World" arXiv 2105.10919. Meta-World 기반 연속 RL 벤치마크, 코드는 프로젝트 페이지에 있음 [P] | https://arxiv.org/abs/2105.10919 | 2026-10-16 | VERIFIED |
| Hypernetwork로 조건화한 RL로 동역학이 다른 고정익 기체군의 정책 학습, arXiv 2604.03392. 엔진 A/B 조건화 대안이 될 수 있음 [P] | https://www.alphaxiv.org/abs/2604.03392 | 2026-10-16 | VERIFIED (초록 일부만 확인) |
| Peyvan, Kumar, Karniadakis, "Fusion-DeepONet", arXiv 2501.01934. 수렴-확산(CD) 노즐 초음속 유동에서 MeshGraphNet보다 우수. 데이터/코드 공개 여부는 확인 못함 [P] | https://arxiv.org/abs/2501.01934 | 2026-10-16 | VERIFIED |
| Roohi & Mahdavi, "Shock-Aware Physics-Guided Fusion-DeepONet for Rarefied Micro-Nozzle Flows", arXiv 2510.17887 (DSMC, 희박 유동이라 본 프로젝트와 영역이 다름) [P] | https://arxiv.org/abs/2510.17887v1 | 2026-10-16 | VERIFIED |
| Waxenegger-Wilfing et al. (DLR), "ML Methods for Design and Operation of Liquid Rocket Engines", arXiv 2102.07109. 모델링, 제어, 상태감시 개관. 구체 기법은 초록에 없음 [P] | https://arxiv.org/abs/2102.07109 | 2026-10-16 | VERIFIED |
| HF `taarg/MultiFidelity_Nozzle_JetFlows`: Napoli Federico II 제작. 파라메트릭 노즐 형상 3개 변수. RANS(SU2) 92건, LES(CharLES) 18건, 배열 크기 (N,12,128,128), Apache-2.0 [OSS 데이터] | https://huggingface.co/datasets/taarg/MultiFidelity_Nozzle_JetFlows | 2026-10-16 | VERIFIED |
| HF `JinglaiZheng/Laval_nozzle`: train h5 약 0.9 GB, test h5 약 0.22 GB, CC-BY-4.0. README에 설명과 스키마가 없음 [OSS 데이터] | https://huggingface.co/datasets/JinglaiZheng/Laval_nozzle | 2026-10-16 | VERIFIED (내용은 UNKNOWN) |
| HF `pdebench/1D-Compressible-Navier-Stokes`, CC-BY-4.0 (FNO 파이프라인 점검용 압축성 벤치마크) [OSS 데이터] | https://huggingface.co/datasets/pdebench/1D-Compressible-Navier-Stokes | 2026-10-16 | VERIFIED |
| EmbersArc/gym-rocketlander: MIT, Box2D, 바지선 착륙, 이산/연속 행동, 306 stars, 커밋 3개, 2D(상태변수로 추정) [OSS] | https://github.com/EmbersArc/gym-rocketlander | 2026-10-16 | VERIFIED (2D는 INFERRED) |
| jiupinjia/rocket-recycling: 2D 강체, PPO+GAE, hover와 landing 과제, **CC BY-NC-SA 4.0**, 302 stars [OSS] | https://github.com/jiupinjia/rocket-recycling | 2026-10-16 | VERIFIED |
| Tuxliri/RL_Rocket_6DOF: 6-DoF gym 환경, 3D 렌더링, wandb 연동, MSc 논문용, **LICENSE 없음**, 3 stars [OSS] | https://github.com/Tuxliri/RL_Rocket_6DOF | 2026-10-16 | VERIFIED |
| dhairya-pandya/RocketLandingWithRL: MIT, 2D 움직이는 드론십, PPO/SAC/TD3/GRPO/ES/DQN, PID와 CEM 기준선, 커리큘럼 모듈 포함 [OSS] | https://github.com/dhairya-pandya/RocketLandingWithRL | 2026-10-16 | VERIFIED |
| GitHub 저장소들의 마지막 커밋 날짜(페이지에 공란, gh API 접근 차단) | — | 2026-10-16 | UNKNOWN |
| NASA 3D Resources: "free and without copyright", 357개 모델, GitHub 미러 3.8k stars [O] | https://nasa3d.arc.nasa.gov ; https://github.com/nasa/NASA-3D-Resources | 2026-10-16 | VERIFIED |
| NASA 이용지침: 미디어는 대체로 미국 내 저작권 없음. **NASA Insignia/Logotype은 공공 영역이 아님**. 보증 암시 금지. **AI 학습이나 AI 생성 이미지에 휘장 사용 불가** [O] | https://nasa.gov/nasa-brand-center/images-and-media | 2026-10-16 | VERIFIED |
| NASA Saturn V GLB(905 KB, 2025-04), Mobile Launcher GLB(assembled와 pad only), Apollo LM GLB(약 700 KB). 각 페이지에 개별 라이선스 문구 없음 [O] | https://science.nasa.gov/3d-resources/saturn-v/ ; https://science.nasa.gov/3d-resources/mobile-launcher/ ; https://science.nasa.gov/3d-resources/apollo-lunar-module/ | 2026-10-16 | VERIFIED |
| Isaac Sim 에셋은 "Isaac Sim Additional Software and Materials License" 적용. 라이선스가 명시 허용하지 않는 수정·재배포 금지. 로켓이나 착륙장 에셋은 목록에 없음 [O] | https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-faq.html ; https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_overview.html | 2026-10-16 | VERIFIED |
| Sketchfab "SpaceX Starship-With landing legs deployed"(AllThingsSpace): CC BY 4.0, 약 343k tris. 상표(SpaceX) 문제 있음 [C] | https://sketchfab.com/3d-models/spacex-starship-with-landing-legs-deployed-168d98f7b4d747f88d301050ae645560 | 2026-10-16 | VERIFIED |
| BlendSwap "Falcon 9 Reusable (F9R)": 라이선스 표기가 CC-BY와 CC0로 서로 충돌함 [C] | https://www.blendswap.com/blend/12111 | 2026-10-16 | VERIFIED (라이선스는 UNKNOWN) |

## 후보 자산
| 이름 | URL | 버전 | 라이선스 | 용도 | 채택 권고 |
|---|---|---|---|---|---|
| Gaudet 2018 (S10) | https://arxiv.org/abs/1810.08719 | v1 | arXiv | 6-DoF PPO 보상 설계와 이중 할인율 참고 | 채택(참고문헌) |
| Souza 2022 (S15) | https://sol.sbc.org.br/index.php/sbrlars/article/view/23169 | SBR/LARS 2022 | SBC 학회 논문 | TVC 행동공간 설계 참고(3-DoF) | 채택. 인용 시 "3-DoF"로 정정 |
| Iafrate 2025 | https://doi.org/10.1016/j.actaastro.2024.11.028 | Acta Astronautica 227 | Elsevier | 6-DoF PPO 1단 착륙. 가장 가까운 선행연구 | 채택(핵심 참고) |
| Carradori 2024 thesis | https://repository.tudelft.nl/record/uuid:bf2a598c-9694-40cd-8dec-03b73d539b54 | 2024-09 | TU Delft repo | 6-DoF 메타 RL, 순환 정책(GTrXL) | 참고 |
| RAJS (S11) | https://arxiv.org/abs/2407.15083 | v1 | arXiv | curriculum과 jump-start | 채택(이미 검증됨) |
| Wołczyk 2024 fine-tune/forgetting | https://arxiv.org/abs/2402.02868 | v3 | arXiv | 엔진 A에서 B로 fine-tune할 때 BC와 KS 정규화 기준선 | 채택(핵심) |
| EWC (Kirkpatrick) | https://arxiv.org/abs/1612.00796 | v2 | arXiv/PNAS | 비교 기준선 | 보조 기준선으로 채택 |
| Continual World | https://arxiv.org/abs/2105.10919 | v3 | arXiv(코드는 프로젝트 사이트) | 연속 RL 평가 지표(forgetting, forward transfer) 설계 | 지표 설계 참고 |
| PhysicsNeMo darcy_fno (S4) | https://github.com/NVIDIA/physicsnemo | latest(버전 미확인) | Apache-2.0 | FNO 학습 파이프라인 템플릿 | 채택(FNO 단계) |
| Fusion-DeepONet | https://arxiv.org/abs/2501.01934 | v2 | arXiv | CD 노즐 neural operator 비교 대상 | 참고(코드 공개 여부 확인 필요) |
| MultiFidelity_Nozzle_JetFlows | https://huggingface.co/datasets/taarg/MultiFidelity_Nozzle_JetFlows | 2025-11-13 수정 | Apache-2.0 | 노즐 형상에서 제트 유동장으로 가는 surrogate 실험(외부 제트 유동) | 조건부 채택(보조 벤치마크) |
| Laval_nozzle (HF) | https://huggingface.co/datasets/JinglaiZheng/Laval_nozzle | 2025-04-26 | CC-BY-4.0 | 라발 노즐 FNO/MLP 데이터 후보 | 보류(스키마 확인 후) |
| PDEBench 1D CNS | https://huggingface.co/datasets/pdebench/1D-Compressible-Navier-Stokes | 2024-02 | CC-BY-4.0 | 압축성 유동 FNO 파이프라인 sanity check | 선택 |
| EmbersArc/gym-rocketlander | https://github.com/EmbersArc/gym-rocketlander | 커밋 3개(날짜 UNKNOWN) | MIT | 2D 스모크 테스트와 보상 참고 | 참고(2D) |
| jiupinjia/rocket-recycling | https://github.com/jiupinjia/rocket-recycling | 커밋 15개 | CC BY-NC-SA 4.0 | 2D PPO hover/landing 참고 | 코드 복사 금지(비상업·동일조건). 아이디어만 참고 |
| dhairya-pandya/RocketLandingWithRL | https://github.com/dhairya-pandya/RocketLandingWithRL | 커밋 68개 | MIT | 2D 커리큘럼과 다중 알고리즘 기준선 | 참고 |
| Tuxliri/RL_Rocket_6DOF | https://github.com/Tuxliri/RL_Rocket_6DOF | 커밋 247개 | 라이선스 없음 | 6-DoF gym 구조 참고 | 코드 재사용 금지. 설계만 참고 |
| NASA Mobile Launcher GLB (pad only) | https://science.nasa.gov/3d-resources/mobile-launcher/ | 2025-04-11 | 저작권 없음(NASA 지침, 휘장 제외) | 착륙장/발사대 시각화 | 채택 후보. GLB→USD 변환, 로고 제거 |
| NASA Saturn V / Apollo LM GLB | https://science.nasa.gov/3d-resources/saturn-v/ | 2025-04 | 저작권 없음(동일 조건) | 시각화 플레이스홀더 | 선택. 형상이 VTVL과 맞지 않음 |
| Sketchfab Starship with legs | https://sketchfab.com/3d-models/spacex-starship-with-landing-legs-deployed-168d98f7b4d747f88d301050ae645560 | 2022-04 | CC BY 4.0 | VTVL 외형 시각화 | 비권고(SpaceX 상표, 343k tris) |
| Isaac Sim 기본 에셋 | https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_overview.html | 6.2.0 | NVIDIA 에셋 라이선스(재배포 제한) | 지면과 조명 | 로컬 사용만. 저장소 커밋 금지 |
| 자체 USD 프리미티브(실린더 본체, 짐벌 노즐, 4개 다리, 원형 패드) | — | — | 프로젝트 자체 | 물리 충돌체와 시각 모델 | **권고(기본안)** |

## 미해결
- S15의 DOI가 SBC 페이지에 없음. Crossref에서 SBC 논문 DOI(10.5753/...)가 있는지 확인 필요. 원문 PDF로 "advanced RL technique" 비교 결과를 확인하지 못함.
- GitHub 저장소들의 마지막 커밋 날짜: 세션에서 gh/API 접근이 차단되고 페이지에도 날짜가 비어 있음. 저장소 접근 권한이 있는 세션에서 다시 확인해야 함.
- arex18/rocket-lander, Raiqueeee/SpaceX-Falcon9-Landing, AxelBoularot/6DDL_TVC(6-DoF TVC 모델로켓 시뮬레이터, 실비행 로그 비교라고 표기됨)는 검색 결과에서만 확인했고 라이선스는 미확인.
- Laval_nozzle 데이터의 스키마와 출처 논문을 모름(README 비어 있음). h5 헤더를 확인해야 함. 로켓 노즐 Isp/추력 라벨이 있는 공개 데이터셋은 HF와 Zenodo 검색에서 찾지 못함.
- Fusion-DeepONet의 노즐 데이터가 공개됐는지 미확인(원문 Data Availability 절 확인 필요).
- 실제 TVC 로켓형 기체에서 sim-to-real RL을 한 논문은 2022-2026 범위에서 찾지 못함(QuadRocket은 RL이 아님). 이 갭은 프로젝트 범위 밖의 근거 공백으로 기록함.
- NASA GLB 개별 페이지에는 라이선스 문구가 없음. "without copyright" 사이트 문구와 NASA 휘장 제한(AI 학습에 휘장 사용 금지 포함)을 함께 적용한다는 해석은 INFERRED임.
- PhysicsNeMo 정확한 릴리스 버전은 미확인(Releases 섹션이 비어 보임). PyPI(`nvidia-physicsnemo`)로 확인 필요.
