# Track 4 — Flight Dynamics & Control

## 결론
- **동역학 기준 모델(a)**: RocketPy는 의존 라이브러리로 쓰지 않고 교차검증용으로만 둔다. 자체 Python 6-DoF를 만든다. 상태 벡터는 14개(pos_w, vel_w, q, ω_b, m_prop)로 하고, CoM 기준 Newton–Euler 방정식에 가변 질량, CoM, 관성을 반영한다. 자세는 q̇ = ½ q⊗[0, ω] (Hamilton)로 적분한다. 적분기는 **고정 스텝 RK4**로 한다. dt는 1–5 ms로 두고 제어·밸브 이벤트 주기와 맞춘다. 매 스텝 끝에 쿼터니언을 정규화한다. 기준 해를 만들 때만 scipy `solve_ivp`(DOP853/LSODA, 엄격한 허용오차)를 쓴다.
- **jet damping과 İ 항**: P0에서는 끄되 플래그로 남긴다. RocketPy 1.13 `u_dot`에는 이 항들이 들어 있다. NASA TR R-115는 피치/요 jet damping이 "typical하게 크다"고 한다. 따라서 P1에서는 RocketPy와 교차검증해 그 크기를 정량화한다.
- **제어 툴체인(b)**: PD/PID는 직접 구현한다. LQR과 선형화 검증은 **python-control 0.10.2**(BSD-3)로 한다. MPC 프로토타입은 **CasADi 3.8.1**(LGPL) 위의 **do-mpc 5.1.2**(LGPL)로 만든다. 실시간 NMPC가 필요해지면 **acados**(BSD-2, 소스 빌드)로 옮긴다. 향후 유도(guidance) 베이스라인은 G-FOLD/LCvx(Açıkmeşe & Ploen 2007, Blackmore 2010)이고 cvxpy로 구현한다.
- **RCS 할당(c)**: P0는 구성 행렬 B(6×N)의 rank와 부호 커버리지를 검사하는 것부터 시작한다(롤 권한을 확인해야 한다). 이어서 phase-plane 데드밴드와 jet-select(LUT/LP)를 구현한다. 연속 토크 명령은 pseudo-inverse(재분배) 후 PWM/PWPF와 최소 임펄스 비트(MIB)를 적용해 on/off로 바꾼다. P1에서는 `scipy.optimize.linprog`(simplex/HiGHS) 기반 LP 할당과 attainable moment set 분석을 더한다(Bodson 비교 논문, NASA Hall et al.).
- RocketPy의 제어 기능은 air brakes용 `controller_function`(6–8 인자, sampling_rate)뿐이다. TVC, RCS, 능동 착륙 루프는 없다. 이 점을 1.13.0 소스 grep으로 확인했다(VERIFIED).

## 근거 표
| 주장 | 출처 URL | 확인 일자 | 상태(VERIFIED/INFERRED/UNKNOWN) |
|---|---|---|---|
| RocketPy 최신 버전은 1.13.0(2026-07-22 업로드)이고 MIT 라이선스, requires_python ≥3.10, 순수 py3 wheel이다 | https://pypi.org/project/rocketpy/ (PyPI JSON API) | 2026-10-16 | VERIFIED |
| RocketPy Flight 기본값은 "Standard 6-DOF equations"이고 기본 ode_solver는 LSODA다 | https://docs.rocketpy.org/en/latest/user/flight.html | 2026-10-16 | VERIFIED |
| RocketPy는 가변 질량을 반영한다. `u_dot`에서 mass_flow_rate를, `u_dot_generalized`에서 total_mass_flow_rate와 그 미분을 사용한다(1.13.0 sdist의 flight.py) | https://pypi.org/project/rocketpy/1.13.0/ (sdist의 rocketpy/simulation/flight.py L1905–2507) | 2026-10-16 | VERIFIED |
| RocketPy `u_dot`에는 jet damping 형태의 항이 있다(`mass_flow_rate*nozzle_radius**2/2`, `2*c*mass_flow_rate*omega`) | 위 sdist의 flight.py L2140–2198 | 2026-10-16 | VERIFIED |
| RocketPy 제어는 `_Controller` 클래스와 `Rocket.add_air_brakes(controller_function, sampling_rate, …)`로 구성된다. 시그니처는 6/7/8 인자다. sampling_rate=None이면 연속 제어기다 | https://docs.rocketpy.org/en/latest/user/controllers.html , https://docs.rocketpy.org/en/latest/user/airbrakes.html | 2026-10-16 | VERIFIED |
| RocketPy에는 TVC, gimbal, RCS 기능이 없다. "thrust vector"는 정적 편심(misalignment) 파라미터로만 나온다(rocket.py L126–129, L2060) | 1.13.0 sdist grep 결과와 위 문서 페이지 | 2026-10-16 | VERIFIED |
| NASA NTRS 20160001840은 열람 가능하다. Hall, Hough, Orphee, Clements, "Design and Stability of an On-Orbit Attitude Control System Using Reaction Control Thrusters"(AIAA GNC 2016, M15-4620)다. phase-plane 설계와 describing function 안정성, jet selection, 필터, automaneuver를 다룬다. 할당 이론 논문은 아니다 | https://ntrs.nasa.gov/citations/20160001840 | 2026-10-16 | VERIFIED |
| Bodson, "Evaluation of Optimization Methods for Control Allocation"(JGCD 2002)는 redistributed pseudo-inverse, QP, fixed-point, direct allocation(Durham AMS, simplex LP), mixed L1 LP를 비교한다. 결론은 LP/최적화가 실시간으로 쓸 수 있고 feasible 명령을 보장한다는 것이다 | https://my.eng.utah.edu/~bodson/code/Evaluation%20of%20optimization%20methods.pdf | 2026-10-16 | VERIFIED(학술지명과 연도는 검색 결과 기준, INFERRED) |
| Martz, NASA TR R-115(1961)는 피치/요 jet damping 효과가 "typical"하게 크다고 한다. 질량 불균형 wobble에는 영향이 작다고 한다 | https://ntrs.nasa.gov/api/citations/20190002211/downloads/20190002211.pdf | 2026-10-16 | VERIFIED |
| 저속·단시간 VTVL hop(스핀이 작은 경우)에서는 P0 단계에서 jet damping을 생략해도 된다 | 위 근거로부터 추론 | 2026-10-16 | INFERRED |
| Solà, "Quaternion kinematics for the error-state Kalman filter"(arXiv:1711.02508, 2017)는 쿼터니언 미분과 적분을 다룬다 | https://arxiv.org/abs/1711.02508v1 | 2026-10-16 | VERIFIED(초록만 확인했다. q̇ 식과 Hamilton 규약은 본문에서 확인해야 한다) |
| 6-DoF 가변 질량 강체 교과서로 Zipfel의 *Modeling and Simulation of Aerospace Vehicle Dynamics*(AIAA)와 Stevens & Lewis의 *Aircraft Control and Simulation*을 표준 문헌으로 쓸 수 있다 | (이번에 fetch하지 않았다) | 2026-10-16 | UNKNOWN(서지 확인 필요) |
| python-control 최신 버전은 0.10.2(2025-07-05)이고 BSD-3-Clause, requires ≥3.10이다. 순수 py3 wheel이고 LQR과 MPC/RHC, 최적제어를 제공한다 | https://pypi.org/project/control/ , https://python-control.readthedocs.io/en/latest/ | 2026-10-16 | VERIFIED |
| python-control의 Python 3.13 지원은 classifier에 3.12까지만 표기되어 있다. 순수 Python이라 3.13에서도 동작할 것으로 본다 | PyPI JSON | 2026-10-16 | INFERRED |
| CasADi 3.8.1(2026-09-16)은 LGPLv3+다. `cp311-abi3` wheel이 있어 3.13에서 설치할 수 있다(stable ABI) | https://pypi.org/project/casadi/ | 2026-10-16 | VERIFIED(wheel 태그 기준). 실제 3.13 import는 INFERRED |
| do-mpc 5.1.2(2026-09-03)는 LGPLv3이고 casadi≥3.6.0에 의존한다. NMPC, multi-stage robust MPC, MHE를 지원하며 순수 py3다 | https://pypi.org/project/do-mpc/ , https://www.do-mpc.com/en/latest/ | 2026-10-16 | VERIFIED |
| acados는 2-Clause BSD 라이선스다. C 코어를 CMake로 빌드해야 하고 Python 인터페이스 `acados_template`은 PyPI에 없다(소스 설치). CasADi로 모델을 기술한다 | https://github.com/acados/acados/blob/main/LICENSE , https://docs.acados.org/installation/index.html | 2026-10-16 | VERIFIED |
| cvxpy 1.9.3(2026-09-19)은 Apache-2.0이고 requires ≥3.11이다 | https://pypi.org/project/cvxpy/ | 2026-10-16 | VERIFIED |
| Blackmore, Açıkmeşe, Scharf, "Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization", JGCD 33(4), 2010, doi:10.2514/1.47202 | https://ece.uwaterloo.ca/~ece602/Projects/2016/Papers/paper9.pdf | 2026-10-16 | VERIFIED |
| Açıkmeşe & Ploen 2007(JGCD, Mars PDG convex programming)와 Açıkmeşe, Carson, Blackmore 2013(IEEE TCST, lossless convexification)은 G-FOLD 구현 README에 인용되어 있다 | https://github.com/Wrg1t/G-FOLD | 2026-10-16 | VERIFIED(인용 사실). DOI는 UNKNOWN |
| Malyuta et al., "Convex Optimization for Trajectory Generation"(arXiv:2106.09125)은 LCvx, SCvx, GuSTO 튜토리얼이다 | https://arxiv.org/abs/2106.09125 | 2026-10-16 | VERIFIED |
| Szmuk & Açıkmeşe, "Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time"(arXiv:1802.03827, doi:10.2514/6.2018-0617)는 단일 짐벌 엔진 6-DoF 강체를 다룬다 | https://arxiv.org/abs/1802.03827 | 2026-10-16 | VERIFIED |

## 후보 자산
| 이름 | URL | 버전 | 라이선스 | 용도 | 채택 권고 |
|---|---|---|---|---|---|
| RocketPy | https://github.com/RocketPy-Team/RocketPy | 1.13.0 (2026-07-22) | MIT | 6-DoF 가변 질량 교차검증, jet damping 항 참조 | 교차검증 전용(TVC/RCS 없음) |
| python-control | https://github.com/python-control/python-control (pip `control`) | 0.10.2 | BSD-3-Clause | LQR/dlqr, 선형화, 안정여유 분석 | **채택(P0)** |
| CasADi | https://web.casadi.org/ (pip `casadi`) | 3.8.1 | LGPL-3.0+ | 심볼릭 모델, NLP, MPC 백엔드 | **채택(P1)**, 동적 링크로 사용 |
| do-mpc | https://www.do-mpc.com/ (pip `do-mpc`) | 5.1.2 | LGPL-3.0 | MPC/MHE 프로토타입 | **채택(P1 프로토타입)** |
| acados | https://github.com/acados/acados (소스 빌드, `acados_template`) | 릴리스 미확인 | BSD-2-Clause | 실시간 NMPC(RTI) | 보류(P2, 빌드 부담) |
| cvxpy | https://www.cvxpy.org/ (pip `cvxpy`) | 1.9.3 | Apache-2.0 | G-FOLD/LCvx SOCP | 채택(유도 베이스라인용) |
| scipy (`solve_ivp`, `linprog`) | https://scipy.org/ | 미확인 | BSD-3-Clause | 기준 적분, LP 기반 RCS 할당 | 채택 |
| Wrg1t/G-FOLD | https://github.com/Wrg1t/G-FOLD | 15 commits, 날짜 미표시 | MIT | Python 3 LCvx/G-FOLD 참조 구현 | 참조(알고리즘 대조) |
| samutoljamo/g-fold (pip `gfold`) | https://github.com/samutoljamo/g-fold | 0.3.2 (2026-06-29) | MIT | Rust 코어(Clarabel) + Python 바인딩 G-FOLD | 참조/후보(빠른 SOCP) |
| jonnyhyman/G-FOLD-Python | https://github.com/jonnyhyman/G-FOLD-Python | 날짜 미표시, Python 2.7 | GPL-3.0 | G-FOLD 구현 | 비권고(GPL, Py2) |
| EmbersArc/SCvx | https://github.com/EmbersArc/SCvx | 날짜 미표시 | MIT | 6-DoF 로켓 착륙 SCvx (cvxpy) | 참조(6-DoF 착륙 궤적 교차검증) |
| PyFlyt (Rocket-Landing-v3) | https://github.com/jjshoots/PyFlyt | 0.29.0 (2025-03-01) | MIT | PyBullet 기반 로켓 착륙 환경(커뮤니티 OSS) | 참조(RL/시나리오 비교, 물리 상세 미확인) |
| govindchari/6DOF-Model-Rocket-Simulation | https://github.com/govindchari/6DOF-Model-Rocket-Simulation | 4 commits | 라이선스 없음 | MATLAB/Simulink TVC + 쿼터니언 PID | 아이디어 참고만(라이선스 없음, Python 아님) |

## 미해결
- GitHub API/clone이 이 세션에서 차단되어 있었다. 그래서 Wrg1t/G-FOLD, EmbersArc/SCvx, G-FOLD-Python, acados의 최근 커밋 날짜와 acados 최신 릴리스 태그를 확인하지 못했다(UNKNOWN).
- Zipfel과 Stevens & Lewis 교과서의 서지, 그리고 Açıkmeşe & Ploen 2007과 Açıkmeşe/Carson/Blackmore 2013의 DOI는 직접 fetch하지 못했다.
- python-control과 CasADi가 Python 3.13에서 실제로 import되고 테스트를 통과하는지는 실행 검증이 필요하다(classifier에는 3.12까지만 표기됨).
- 짐벌을 단순화했을 때 jet damping과 İ(관성 변화율) 항의 크기를 정량화해야 한다. RocketPy와 같은 조건으로 돌려 비교하는 실험을 Track 테스트 계획에 넣어야 한다.
- PyFlyt Rocket 모델에 짐벌, RCS, 가변 질량이 있는지는 확인하지 못했다(문서 하위 페이지를 열지 않았다).
- RCS 할당 문헌 중 Durham(1993) attainable moment set 원 논문과 Wie의 *Space Vehicle Dynamics and Control*(PWPF)은 이번에 fetch하지 않았다(UNKNOWN).
- NTRS 20160001840은 citation 페이지에서 확인했다. PDF 본문의 jet-select 알고리즘 상세는 직접 읽지 않았다.
