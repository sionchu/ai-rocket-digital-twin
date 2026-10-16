# ADR-0001 툴체인 버전 고정

- 상태: 채택 (2026-10-16)
- 근거: `docs/research/01`–`05`

## 결정
| 영역 | 고정 버전 | 이 환경에서 실제 사용 | 비고 |
|---|---|---|---|
| Python | 3.11 이상 (Isaac 환경은 3.11) | 3.13.16 | 참조 모델은 3.11–3.13 공통 문법만 사용 |
| numpy / scipy / pyyaml | 2.5.3 / 1.18.1 / 6.0.1 | 같음 | `pyproject.toml` |
| 열화학 | NASA `cea==3.3.4` (Apache-2.0) | 미설치 (P0는 이상기체 1D) | 단위 변환은 어댑터 한 곳에서 (bar→Pa, m/s→s) |
| CFD | OpenFOAM ESI **v2606** (`rhoCentralFoam`) | **v1912** (Ubuntu 24.04 아카이브 `openfoam 1912.200626`) | openfoam.com/.org 저장소가 네트워크 정책으로 403. 결과 manifest에 실제 버전 기록 |
| 시뮬레이션·RL | Isaac Lab **v2.3.2** + Isaac Sim **5.1.0** + PhysX + `rsl-rl-lib==3.1.2` | 없음 (GPU 없음, BLOCKED) | 3.0 GA 나오면 재평가 (`--load_run`/`--resume` 제거, positions 기준점 변경) |
| 제어 도구 | python-control 0.10.2, CasADi 3.8.1, do-mpc 5.1.2 | 미설치 | PD/할당은 자체 구현 (scipy) |
| 비교 시뮬레이터 | RocketPy 1.13.0 (MIT) | 미설치 | TVC/RCS 없음 확인됨 (소스 grep). 교차검증 전용 |

## 결과
- v1912 결과는 v2606 재현 전까지 "v1912 결과"로 표기한다. 솔버 알고리즘(Kurganov–Tadmor)은 같지만 기본값 차이는 확인하지 않았다.
- RocketCEA·CEA_Wrap(GPL)은 기본 의존성에서 뺀다.
