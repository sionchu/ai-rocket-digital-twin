# ADR-0006 P2 CFD 설정

- 상태: 채택 (v1912로 실행, v2606 재현은 미실행)

- 영역: 연소실 입구부터 노즐 출구면까지 내부 유동만. 2D 축대칭 wedge(±2.5°, 1셀 두께, 축 면 붕괴). blockMesh 2블록(수렴부/확산부), 벽은 `geometry.build_contour`의 polyLine.
- 물리: 비점성(μ=0), 열량적 완전기체(γ=1.2, M=22 g/mol), slip 벽, 입구 totalPressure/totalTemperature, 출구 waveTransmissive(초음속).
- 비점성 완전기체 해는 T0 고정 시 Pc에 비례하므로, 노즐이 가득 흐르는(attached) 동안 F(thr, Pa) = thr·S_nom − Pa·Ae로 한 번의 공칭 실행에서 전체 맵을 만든다. 분리는 모델링하지 않으며, 1D와 같은 Pe/Pa < 0.4 규칙으로 invalid 처리한다.
- 격자 3단계, 비율 1.5 (GCI 조건 r > 1.3), Celik et al. 2008 절차.
- 초기장: 1D 등엔트로피 해 (수렴 시간 단축).
- maxCo 0.4: rhoCentralFoam의 KT Courant 계산이 wedge 면의 c|Sf|를 포함해 축 근처 셀에서 과대평가된다(관찰: 평균 Co 0.02, 최대 0.2). 안정성은 QA(유한값, 수렴)로 확인한다.
- v1912 Debian 빌드는 function object 사전 처리에서 `IOstream "sha1"` 오류가 난다. 출구면 적분은 `cfd/post.py`가 기록된 필드와 polyMesh에서 직접 계산한다.
- 점성·열전달·연소·분리는 범위 밖(P2 MVP). 결과는 "inviscid CFD"로 표기한다.
