# Track 2 — CFD & Geometry

## 결론
- **솔버: ESI/Keysight OpenFOAM v2606 (openfoam.com, GPL, 2026-06)을 채택하고 `rhoCentralFoam`(Kurganov–Tadmor central-upwind)을 그대로 쓴다.** openfoam.org(v13 2025-07, v14 2026-07-14)에서는 v11부터 rhoCentralFoam이 `foamRun -solver shockFluid` 모듈로 대체됐다. 설계 문서의 솔버 이름을 바꾸지 않으려면 ESI 계열이 맞다. openfoam.org v14의 shockFluid는 백업 옵션으로 둔다.
- **축대칭은 wedge 메쉬로 구현한다.** 1셀 두께, 각도 5° 이하(예: ±2.5°), 앞뒤 면은 각각 `wedge` 패치이고 축 위의 면은 붕괴(collapsed)시킨다. 노즐 메쉬는 Python으로 contour 점을 만든 뒤 blockMesh 하나로 만드는 방식을 1순위로 권고한다(정렬 격자에 3단계 이상 격자를 쉽게 만들 수 있음). Gmsh 4.15.2 → `gmshToFoam`(ASCII msh 전용)은 2순위다.
- **검증 케이스**
  - (a) NASA NPARC CDV 축대칭 cosine 노즐: 비점성, 배압 3개 중 하나에 normal shock이 있어 shock 민감도 게이트에 쓴다. 비교 대상은 해석해다.
  - (b) Back–Massier–Gier(JPL 1965) 30°/15° 원추 노즐: 벽면 압력 실험값이 있어 conical_A에 대응한다.
  - bell_B는 공개 실험 데이터를 확인하지 못했다. 1D 등엔트로피 해 + GCI로 검증한다.
- **후처리:** PyVista 0.49.0(MIT)의 `OpenFOAMReader`(.foam), ParaView 6.2.0, 필요 시 `foamToVTK`를 쓴다. 형상은 CadQuery 2.8.0(Apache-2.0)으로 시각화 자산(STEP/STL)만 만들고, CFD 유체 도메인은 같은 contour CSV에서 따로 생성한다. 격자 수렴은 Celik et al.(ASME JFE) 5단계 GCI 절차를 따른다(r>1.3, 계수 1.25).

## 근거 표
| 주장 | 출처 URL | 확인 일자 | 상태(VERIFIED/INFERRED/UNKNOWN) |
|---|---|---|---|
| OpenFOAM v2606는 2026-06에 나온 ESI(현 Keysight 배포) 최신 릴리스이고 GPL이다. Ubuntu 24.04/22.04 패키지를 제공한다 | https://www.openfoam.com/news/main-news/openfoam-v2606 | 2026-10-16 | VERIFIED |
| v2606 문서에 rhoCentralFoam이 compressible 솔버로 있고 Kurganov–Tadmor central-upwind를 쓴다 | https://doc.openfoam.com/2606/tools/processing/solvers/rtm/compressible/rhoCentralFoam | 2026-10-16 | VERIFIED |
| ESI 저장소 tutorials/compressible/rhoCentralFoam에는 LadenburgJet60psi, biconic25-55Run35, forwardStep, movingCone, obliqueShock, shockTube, wedge15Ma5가 있다 | https://gitlab.com/openfoam/core/openfoam/-/tree/master/tutorials/compressible/rhoCentralFoam (GitLab API로 조회) | 2026-10-16 | VERIFIED |
| LadenburgJet60psi는 축대칭 wedge 부족팽창 제트 케이스다 | 위 디렉터리(내용은 열어보지 않음) | 2026-10-16 | INFERRED |
| ESI 저장소가 develop.openfoam.com에서 gitlab.com/openfoam/core/openfoam으로 옮겨졌다 | https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian | 2026-10-16 | VERIFIED |
| Docker Hub `opencfd/openfoam-default`에는 2512(2026-01, 압축 약 478 MB), 2506(약 474 MB) 태그가 있고 2606 태그는 없다 | https://hub.docker.com/v2/repositories/opencfd/openfoam-default/tags | 2026-10-16 | VERIFIED |
| ESI apt 패키지 이름은 `openfoam2606-default`이고, `curl https://dl.openfoam.com/add-debian-repo.sh` 스크립트로 저장소를 추가한다 | 공식 페이지에 명령어가 없음(wiki가 로그인을 요구함) | 2026-10-16 | INFERRED |
| openfoam.org v11(2023-07-11)에서 rhoCentralFoam이 shockFluid 모듈로 대체됐다 | https://openfoam.org/?p=8212 | 2026-10-16 | VERIFIED |
| shockFluid: "density-based solution of compressible flow", Kurganov–Tadmor central-upwind, Greenshields et al. 2010 인용 | https://cpp.openfoam.org/dev/shockFluid_8H_source.html | 2026-10-16 | VERIFIED |
| OpenFOAM 13은 2025-07-08에 나왔고 GPLv3이며 shockFluid 튜토리얼 카테고리(diffuserIntake)가 있다 | https://openfoam.org/release/13/ | 2026-10-16 | VERIFIED |
| OpenFOAM 14는 2026-07-14에 나왔다. 패치는 20260724와 20261008이고 Ubuntu 22.04/24.04/26.04 apt 패키지 이름은 `openfoam14`다 | https://openfoam.org/news/ , https://openfoam.org/news/v14-patch/ | 2026-10-16 | VERIFIED |
| v11부터 응용 솔버가 `foamRun` + 솔버 모듈 구조로 "largely replaced" 되었다 | https://doc.cfd.direct/openfoam/user-guide-v13/solvers-modules | 2026-10-16 | VERIFIED |
| wedge 정의: 작은 각도의 쐐기형, 1셀 두께, 축을 따라 배치, 두 면은 각각 wedge 패치 | https://doc.cfd.direct/openfoam/user-guide-v13/boundaries | 2026-10-16 | VERIFIED |
| NPARC CDV는 축대칭 cosine CD 노즐(A 2.5/1.0/1.5 in²), 비점성이고 배압비 0.89/0.75(normal shock)/0.16이다. 중심선 p·M 데이터는 1D 해석해다 | https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html | 2026-10-16 | VERIFIED |
| Back–Massier–Gier 노즐: 수렴부 30°, 확산부 15°, 목 지름 1.803 in, 확대비 2.68, 출구 M≈2.5, 벽면 정압 구멍 32개, 열유속 측정 | https://ntrs.nasa.gov/api/citations/19650010083/downloads/19650010083.pdf | 2026-10-16 | VERIFIED |
| 같은 노즐이 Ansys 검증 케이스 VMFL044로 쓰인다(3° 섹터, SST, 벽면 압력비를 실험과 비교) | https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/fbu_vm/Hlp_VMFL044.html | 2026-10-16 | VERIFIED |
| Seiner M2.0 가열 제트 노즐(NPARC)은 축대칭 2D 격자와 5° 3D 압출 격자를 쓴다 | https://www.grc.nasa.gov/WWW/wind/valid/seiner/Seiner.html | 2026-10-16 | VERIFIED |
| rhoCentralFoam(KNP) 검증 논문: Kamma et al. 2025, JRAME 13(3)에서 평균 오차가 2% 미만이다. 노즐 케이스는 아니다 | https://ph01.tci-thaijo.org/index.php/jrame/article/view/257949 | 2026-10-16 | VERIFIED |
| RPI 학위논문(Martinus 2025): rhoCentralFoam CD 노즐, NPR 8.78–296, 실험과 비교. 학위논문이며 동료심사 논문이 아니다 | https://dspace.rpi.edu/items/201b4a14-b7ff-484b-9873-ac783060b8a8 | 2026-10-16 | VERIFIED |
| SU2 최신 버전은 8.5.0 "Harrier"이고 LGPL-2.1이다. meson/ninja 빌드 또는 바이너리로 설치한다 | https://github.com/su2code/SU2 , https://github.com/su2code/SU2/releases | 2026-10-16 | VERIFIED(연도 미표시) |
| SU2 `AXISYMMETRIC= YES/NO`는 2D 문제용 옵션이다 | https://raw.githubusercontent.com/su2code/SU2/master/config_template.cfg | 2026-10-16 | VERIFIED |
| SU2 NICFD 노즐 튜토리얼은 2D이고 v7.0.x 기준이며 TROVA 실험(Spinelli 2018)과 비교한다. 축대칭 튜토리얼은 없다 | https://su2code.github.io/tutorials/NICFD_nozzle/ , https://su2code.github.io/tutorials/home/ | 2026-10-16 | VERIFIED |
| cadquery 2.8.0(2026-06-21), Apache-2.0, OCP 바이너리 휠에 의존한다. conda 설치가 더 잘 지원되며 pip 지원 Python 표기가 일관되지 않다 | https://pypi.org/project/cadquery/ | 2026-10-16 | VERIFIED |
| gmsh 4.15.2(2026-03-24), GPLv2+, Linux x86-64 휠(glibc 2.24 이상) | https://pypi.org/project/gmsh/ | 2026-10-16 | VERIFIED |
| gmshToFoam은 ASCII msh만 읽고, prism/hex 방향을 자동으로 뒤집는다(`-keepOrientation`로 끔). physical surface → patch | https://cpp.openfoam.org/dev/gmshToFoam_8C_source.html | 2026-10-16 | VERIFIED |
| pyvista 0.49.0(2026-09-08), MIT, Python 3.10 이상. `OpenFOAMReader`(.foam, vtkOpenFOAMReader 래퍼)는 기본으로 모든 patch 배열을 켠다 | https://pypi.org/project/pyvista/ , https://docs.pyvista.org/api/readers/_autosummary/pyvista.openfoamreader | 2026-10-16 | VERIFIED |
| ParaView 6.2.0이 2026-09-30경 공지됐다 | https://discourse.paraview.org/c/announcements | 2026-10-16 | VERIFIED(날짜는 근사) |
| GCI 5단계 절차(Celik, Ghia, Roache, Freitas): 3개 격자, r>1.3, GCI_fine=1.25·e_a/(r^p−1) | https://qa12.asme.org/wwwasmeorg/media/ResourceFiles/Shop/Journals/JFENumAccuracy.pdf | 2026-10-16 | VERIFIED |
| 출판 정보 J. Fluids Eng. 130(7):078001, 2008 | 위 PDF에 권호가 표기되지 않음 | 2026-10-16 | INFERRED |
| NASA GCI 튜토리얼: Fs는 격자 3개 이상일 때 1.25, 2개일 때 3.0. r≥1.1, 점근 범위 확인은 GCI23/(r^p·GCI12)≈1 | https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html | 2026-10-16 | VERIFIED |
| 4코어 CPU에서 2D wedge 노즐(~1e5 셀 미만)은 수 시간 안에 정상상태에 도달한다 | 경험적 추정 | 2026-10-16 | INFERRED |

## 후보 자산
| 이름 | URL | 버전 | 라이선스 | 용도 | 채택 권고 |
|---|---|---|---|---|---|
| OpenFOAM (ESI/Keysight) | https://www.openfoam.com/download | v2606 (2026-06) | GPL | 주 솔버. rhoCentralFoam, blockMesh, extrudeMesh, gmshToFoam, foamToVTK | **채택**. apt 패키지 openfoam2606-default(INFERRED). Docker 2606 태그가 없어 2512 이미지를 대안으로 둔다 |
| OpenFOAM (Foundation) | https://openfoam.org/news/v14-patch/ | 14 (패치 20261008) | GPLv3 | shockFluid(`foamRun`). 교차 검증용 | 백업. 케이스 사전 구조가 달라 ESI와 섞어 쓰지 않는다 |
| rhoCentralFoam 튜토리얼 (LadenburgJet60psi 등) | https://gitlab.com/openfoam/core/openfoam/-/tree/master/tutorials/compressible/rhoCentralFoam | v2606 master | GPL | 케이스 템플릿. fvSchemes(Kurganov), thermo 설정 출발점 | 채택(템플릿) |
| NPARC CDV 축대칭 노즐 | https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html | archive | 공공(NASA) | 비점성 검증, shock 위치 민감도, 질량보존 게이트 | **채택**. 첫 번째 검증 게이트 |
| Back–Massier–Gier 30°/15° 원추 노즐 | https://ntrs.nasa.gov/api/citations/19650010083/downloads/19650010083.pdf | 1965 | 공공(NASA NTRS) | conical_A 실험 검증(벽면 p/p0) | **채택**. 데이터는 PDF 그림에서 디지타이즈해야 한다 |
| Seiner M2.0 노즐 (NPARC) | https://www.grc.nasa.gov/WWW/wind/valid/seiner/Seiner.html | archive | 공공 | 축대칭 제트·plume 참고 | 선택 |
| SU2 | https://github.com/su2code/SU2 | 8.5.0 Harrier | LGPL-2.1 | 대안 솔버. `AXISYMMETRIC=YES`(소스항 방식) | 보류. 교차 검증용 2순위 |
| SU2 NICFD 노즐 튜토리얼 | https://su2code.github.io/tutorials/NICFD_nozzle/ | v7.0.2 기준 | LGPL-2.1 | 2D 노즐 설정 참고(실험 비교 포함) | 참고만 |
| CadQuery | https://pypi.org/project/cadquery/ | 2.8.0 | Apache-2.0 | 시각화·디지털트윈 형상(STEP/STL, revolve) | 채택. conda/mamba 설치 권장, Python 3.11–3.12 |
| Gmsh (pip `gmsh`) | https://pypi.org/project/gmsh/ | 4.15.2 | GPLv2+ | 비정렬·구조 메쉬 대안. 출력은 `Mesh.MshFileVersion=2.2` ASCII + gmshToFoam | 2순위. 출력 후 wedge 패치 타입 변경(createPatch/foamDictionary)이 필요하다 |
| blockMesh (OpenFOAM 내장) | (OpenFOAM 포함) | v2606 | GPL | contour를 polyLine/spline edge로 넣은 wedge 정렬 격자. 격자 수준 n×{1,1.5,2.25} | **1순위** |
| PyVista | https://pypi.org/project/pyvista/ | 0.49.0 | MIT | Python 후처리(.foam 직접 읽기, 축선·벽면 샘플) | 채택 |
| ParaView | https://discourse.paraview.org/c/announcements | 6.2.0 | BSD-3 (INFERRED) | 시각 검토, paraFoam | 채택(GUI) |
| GCI 절차 (Celik et al., ASME JFE) | https://qa12.asme.org/wwwasmeorg/media/ResourceFiles/Shop/Journals/JFENumAccuracy.pdf | 2008 | 논문 | 격자 3단계 GCI 보고 양식 | **채택** |
| NASA GCI 튜토리얼 | https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html | — | 공공 | 점근 범위 점검 공식 | 채택 |

## 미해결
1. ESI v2606의 Ubuntu apt 정확한 명령어와 패키지 크기를 공식 페이지에서 확인하지 못했다(wiki가 로그인을 요구함). Docker Hub에 2606 태그가 없다(최신은 2512). 실제 설치 단계에서 `apt-cache policy openfoam2606-default`로 확인해야 한다.
2. LadenburgJet60psi 튜토리얼이 wedge 축대칭인지, 실험(Ladenburg) 비교 데이터를 포함하는지 내용을 직접 확인하지 못했다.
3. bell_B(Rao/TOP bell)용 공개 실험 데이터를 확인하지 못했다(DLR/Hagemann 계열은 접근 미확인). 현재 계획은 1D 등엔트로피·MOC + GCI로 검증하는 것이다.
4. Back–Massier–Gier 벽면 압력 데이터는 그림을 디지타이즈해야 하고 불확도가 붙는다. 열가스(메탄올 연소) 조건이라 γ와 물성을 정해야 한다.
5. rhoCentralFoam을 정상상태로 쓸 때의 수렴 기준이 정해지지 않았다. 질량유량 in/out 차이 <0.1%, 벽 p 변화 <1e-4 등은 제안값이며 정상상태용 LTS 옵션도 미확인이다.
6. NPARC CDV의 "experimental" 표기는 실제로는 1D 해석해다(페이지 내부 표기가 불일치).
7. CadQuery를 pip로 설치할 때 Python 버전 표기가 일관되지 않다(3.9–3.12 vs ≥3.11). conda 환경으로 고정해야 한다.
