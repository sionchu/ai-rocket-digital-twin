# Track 1 — Propulsion & Thermochemistry

## 결론
- **P0/P1은 자체 구현한 순수 Python 1D 등엔트로피 노즐을 쓴다.** 이상기체·γ 고정·연소 없음 조건이며, 수식은 NASA Glenn의 thrust-equations-summary 페이지 그대로다. 질량유량 ṁ=A*·pt/√Tt·√(γ/R)·((γ+1)/2)^(-(γ+1)/(2(γ-1))), Ae/A*–Me 관계, Te, pe, Ve=Me√(γRTe), F=ṁVe+Ae(pe−p0)를 쓴다. Isp=F/(ṁg0)이며 g0는 9.80665 m/s²로 고정한다(Glenn 페이지에는 "9.8"로 표기됨).
- **CEA는 P2 이후에 NASA 공식 `cea` 3.3.4로 연결한다.** PyPI 2026-08-28 릴리스, Apache-2.0, `pip install cea`, `import cea`, Python 3.11 이상, cp311–cp314 wheel이 있다. 이 버전을 고정(pin)하고, 결과는 Tc, γ, M, c*를 담은 경계조건 스냅샷(JSON)으로만 노즐 모델에 넘긴다. 단위 주의: 예제 기준 Pc는 **bar**, Isp와 c*는 **m/s**로 출력되므로 어댑터 한 곳에서 SI(Pa, s)로 바꾼다.
- **RocketCEA 1.2.3은 기본 경로에서 제외한다.** GPLv3이고 기본 단위가 English이며, wheel이 Windows용만 있어 Linux에서는 gfortran 빌드가 필요하다. 교차검증용 선택 의존성으로만 둔다. CEA_Wrap(GPL)도 같은 이유로 비권장이다. Cantera 3.2.0(BSD-3)은 평형 계산 교차검증용 후보다.
- **conical_A와 bell_B의 차이는 1D 모델에 보정계수를 곱하는 방식으로 표현한다.** conical_A는 발산계수 λ=(1+cosα)/2로 표현하며 NASA TN D-467이 실험으로 확인했다. bell_B의 λ≈0.98–0.99는 아직 1차 출처로 확인하지 못했으므로 매개변수로 두고 "가정"으로 표기한다. 형상 생성은 pygasflow 1.4.1(GPLv3)의 `CD_TOP_Nozzle`을 참조용으로 쓴다.
- **CEA와 1D 비점성 계산 결과는 이상 상한일 뿐 실제 노즐의 정답이 아니다.** 보고서와 UI 모든 곳에 "ideal/inviscid" 라벨을 강제한다.

## 근거 표
| 주장 | 출처 URL | 확인 일자 | 상태 |
|---|---|---|---|
| NASA CEA 문서 최신 버전 3.3.4 | https://nasa.github.io/cea/ | 2026-10-16 | VERIFIED |
| PyPI `cea` 3.3.4: 2026-08-28 릴리스, Apache-2.0, Python 3.11 이상, cp311–cp314 wheel(Linux glibc 2.27+/2.28+ x86-64, macOS 14+ arm64, macOS 15+ x86-64, Windows x86-64) | https://pypi.org/project/cea/ | 2026-10-16 | VERIFIED |
| 저장소 라이선스 Apache-2.0, `python -m pip install cea`, `import cea`, GitHub Releases 섹션 비어 있음 | https://github.com/nasa/cea | 2026-10-16 | VERIFIED |
| pip 설치는 Python API만 제공한다. CLI와 Fortran 산출물이 필요하면 소스 빌드(gfortran 10 이상, CMake 3.19 이상)가 필요하다. conda 배포는 언급이 없다. Excel 바인딩은 64-bit Windows 전용 beta | https://nasa.github.io/cea/installation.html | 2026-10-16 | VERIFIED |
| 열역학 DB 경로 지정: `cea.init("/path/to/cea/data")` 또는 `CEA_DATA_DIR` 환경변수 | https://pypi.org/project/cea/ | 2026-10-16 | VERIFIED |
| 로켓 예제 API: `cea.Mixture`, `cea.RocketSolver(prod, reactants=reac)`, `cea.RocketSolution(solver)`, `solver.solve(solution, weights, pc, pi_p, subar=, supar=, hc=, iac=True)`. 예제의 pc 단위는 bar, 반응물 온도는 K | https://nasa.github.io/cea/examples/rocket/example8.html | 2026-10-16 | VERIFIED |
| 예제 출력 단위: P는 bar, T는 K, ρ는 kg/m³, h는 kJ/kg, c*와 Isp(해면·진공)는 m/s. 결과 필드에 `gamma_s`, `c_star`, `coefficient_of_thrust`, `Isp`, `Isp_vacuum`, `ae_at`가 있다 | https://nasa.github.io/cea/examples/rocket/example8.html | 2026-10-16 | VERIFIED |
| Python API 전체가 일관되게 bar/kJ 단위를 쓴다(예제에서 추정) | https://nasa.github.io/cea/examples/rocket/example8.html | 2026-10-16 | INFERRED |
| RocketCEA 1.2.3(2026-01-19): GPLv3, Beta, 관리자 1인, wheel은 Windows x86-64 cp37–cp314만 제공, 기본 단위 English(SI 옵션 있음) | https://pypi.org/project/rocketcea | 2026-10-16 | VERIFIED |
| RocketCEA는 Linux와 macOS에서 gfortran 설치가 필요하고, Windows 32-bit Python 3.8 이상은 미지원 | https://rocketcea.readthedocs.io/ | 2026-10-16 | VERIFIED (요구사항이라는 명시적 문장은 없다. 문서에 gfortran 설치 절이 따로 있다는 점에서 판단) |
| RocketCEA를 Linux에서 Python 3.12/3.13으로 sdist 빌드할 때 실제로 성공하는지 | — | — | UNKNOWN |
| CEA_Wrap 2.1.2(2026-09-14): GPL, Linux에서는 Fortran 실행파일을 직접 컴파일해야 함, 입력 압력 기본 단위 psi | https://pypi.org/project/CEA-Wrap/ | 2026-10-16 | VERIFIED |
| Cantera 3.2.0(2025-11-18): BSD-3-Clause, Python 3.10–3.14 | https://pypi.org/project/cantera/ | 2026-10-16 | VERIFIED |
| S13: F=ṁVe+Ae(pe−p0)이며, 이 페이지에는 ṁ 식이 없고 목에서 choked(M=1)라고만 서술. 2024-07-19 갱신 | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-thrust-equation/ | 2026-10-16 | VERIFIED |
| S14: ṁ=A*pt/√Tt·√(γ/R)·((γ+1)/2)^(-(γ+1)/(2(γ-1))). Ae/A*=((γ+1)/2)^(-(γ+1)/(2(γ-1)))·(1+(γ−1)/2·Me²)^((γ+1)/(2(γ-1)))/Me. Te/Tt=(1+(γ−1)/2·Me²)^-1. pe/pt=(…)^(-γ/(γ-1)). Ve=Me√(γRTe). 2024-04-26 갱신. 요청에 적힌 식과 일치한다(Pc=pt, Tc=Tt, At=A*) | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-equations-summary | 2026-10-16 | VERIFIED |
| Isp=F/(ṁg0)=Veq/g0이며, 이 페이지는 g0를 9.8 m/s²로 표기 | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/specific-impulse | 2026-10-16 | VERIFIED |
| NASA SP-8120 "Liquid Rocket Engine Nozzles"(1976-07)는 벨·플러그 노즐 최대성능 윤곽을 다룬다 | https://ntrs.nasa.gov/citations/19770009165 | 2026-10-16 | VERIFIED (초록만 확인, 본문의 수치는 미확인) |
| NASA TN D-467(Farley & Campbell, 1960): 원추 노즐 15°/25°/29° 시험에서 추력비가 "(1+cos α)/2" 추세를 따름. 원추 노즐 검증 데이터셋으로 쓸 수 있다 | https://ntrs.nasa.gov/citations/19980227347 | 2026-10-16 | VERIFIED |
| λ=(1+cosα)/2(구면 유동 가정). 70% 벨은 풀 벨과 발산손실이 비슷하다. Rao 원 논문 서지: Jet Propulsion 28, pp.377–382(1958); ARS J. 31, pp.1488–1494(1961) | https://www.seitzman.gatech.edu/classes/ae6450/nozzle_geometries.pdf | 2026-10-16 | VERIFIED (Georgia Tech 강의자료. 원 출처는 Sutton) |
| Rao 1958 논문 제목과 권호: Jet Propulsion Vol.28 No.6 | https://en.wikipedia.org/wiki/Bell_nozzle | 2026-10-16 | VERIFIED (커뮤니티 출처. DOI는 미확인) |
| α=15°에서 λ≈0.983 | 계산값 | 2026-10-16 | INFERRED |
| Rao 벨 노즐 λ≈0.98–0.99(Sutton 교재 표) | — | — | UNKNOWN (교재 원문을 열람하지 못함) |

## 후보 자산
| 이름 | URL | 버전 | 라이선스 | 용도 | 채택 권고 |
|---|---|---|---|---|---|
| 자체 1D 이상 노즐 모듈 | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-equations-summary | — | (자체) | P0/P1 성능 계산의 기준선(γ, R, Tc, Pc, At, ε 입력) | **채택(P0)** |
| NASA CEA (`cea`) | https://github.com/nasa/cea | 3.3.4 (2026-08-28) | Apache-2.0 | 평형·동결 연소 경계조건(Tc, γ, M, c*). `cea==3.3.4`로 pin하고 `CEA_DATA_DIR`를 명시 | **채택(P2+)** |
| RocketCEA | https://pypi.org/project/rocketcea | 1.2.3 (2026-01-19) | GPLv3 | 레거시 CEA와 교차검증 | 선택(optional extra만 허용, 기본 미포함) |
| CEA_Wrap | https://pypi.org/project/CEA-Wrap/ | 2.1.2 (2026-09-14) | GPL | 레거시 CEA 래퍼 | 비권장 |
| Cantera | https://pypi.org/project/cantera/ | 3.2.0 (2025-11-18) | BSD-3-Clause | 독립적인 평형 계산 교차검증 | 보류(P3 검증용) |
| pygasflow | https://github.com/Davide-sd/pygasflow | 1.4.1 (2025-11-20) | GPLv3 | `CD_Conical_Nozzle`, `CD_TOP_Nozzle`(Rao TOP), `CD_Min_Length_Nozzle`(평면 MOC), `Rao_Parabola_Angles` | 참조·검증용(GPL이므로 코드 통합 대신 별도 스크립트로 비교) |
| evand/nozzle | https://github.com/evand/nozzle | 미표기 | 미확인 | 축대칭 MOC(MLN, Sivells, TIC), Rao 벨, 원추. 브라우저 데모 https://evand.github.io/nozzle/ | 보류(라이선스 확인 전에는 사용 금지) |
| LREKit (`lrekit`) | https://pypi.org/project/lrekit/ | 0.1.1 (2026-06-29) | 미표기 | Rao/TOP Bézier 벨, MOC(실험적). Python 3.10 이상 3.14 미만, JAX 의존 | 비권장(초기 버전, 라이선스 미확인, 의존성 과다) |
| NASA TN D-467 | https://ntrs.nasa.gov/citations/19980227347 | 1960 | 미국 정부 저작물(공개) | conical_A의 λ 추세를 검증하는 실험 데이터 | **채택(검증 레퍼런스)** |
| NASA SP-8120 | https://ntrs.nasa.gov/citations/19770009165 | 1976 | 미국 정부 저작물(공개) | 벨 윤곽 설계 지침 | 채택(문헌 레퍼런스) |

## 미해결
- **CEA Python API의 단위 규약 전체:** Interfaces와 Quickstart 페이지를 가져오는 데 실패했다(fetch 오류). bar, K, kJ/kg, m/s 단위는 example8에서만 확인했고, API 전반에 적용되는지는 INFERRED 상태다. 단위 변환 어댑터에 단위 테스트를 추가해야 한다.
- **NASA CEA의 GitHub Releases와 태그:** Releases 섹션이 비어 있고, 이 세션에서는 gh API 접근이 거부되었다. 버전 고정은 PyPI 3.3.4를 기준으로 한다. conda 배포 여부도 문서에 언급이 없다.
- **RocketCEA를 Linux에서 Python 3.12/3.13으로 설치할 수 있는지:** wheel이 Windows용만 있다. 소스 빌드는 numpy f2py와 gfortran에 의존하며, 3.12에서 distutils가 제거된 영향이 있는지는 실측이 필요하다(코드 실행이 금지되어 확인하지 않음).
- **Rao 벨 λ≈0.98–0.99와 Sutton 교재의 15° 원추 0.983 표:** 교재 원문을 열람하지 못했다. NASA SP-8120 본문 PDF에서 수치를 확인해야 한다.
- **Rao 1958 논문 DOI(AIAA arc):** 검색 결과에서 확인하지 못했다. 쪽수(377–382)는 Georgia Tech 강의자료에서 가져온 것이다.
- **evand/nozzle과 lrekit의 라이선스, evand/nozzle의 마지막 커밋 일자:** 페이지에 표기가 없고 GitHub API에 접근할 수 없었다.
- **cantera 기반 로켓 전용 래퍼(유지보수 중인 것):** 이번 조사에서 찾지 못했다. Cantera 본체만 확인했다.
