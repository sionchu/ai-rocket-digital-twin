# HANDOFF

갱신: 2026-10-16

## 지금 상태
- 단계 0(조사), 단계 1(P0 vertical slice 중 Python 부분), 단계 2(CFD) 일부, 단계 3(PD 기준선)을 진행했다. 게이트별 판정은 `PROJECT_LEDGER.md`.
- 이 컨테이너에는 GPU가 없다. Isaac Sim/Lab, PPO, 파인튜닝은 BLOCKED.
- OpenFOAM은 Ubuntu 아카이브의 ESI v1912로 실행했다(공식 저장소 403). 고정 목표 v2606과 다르다. 사용자 PC WSL에 v2606이 있다(`/usr/lib/openfoam/openfoam2606`, 사용자 보고).

## 다음 액션 하나
사용자 PC(RTX 4070 Ti 12 GB, `C:\dev\IsaacLab`)에서 설치된 Isaac Lab 버전을 먼저 확인하고(목표 v2.3.2 + Isaac Sim 5.1.0) `scripts/isaac_force_smoke.py`를 작성·실행한다. 단일 강체에 `wrench_bridge`로 (F, τ about CoM)을 걸어 Python 참조 적분기와 한 스텝 응답(선가속도, 각가속도)을 비교한다. 이것이 G2의 남은 부분이고, 통과해야 PPO(단계 4)로 넘어간다.

## 열린 질문 (사용자 결정 필요)
1. 착륙 성공 임계값(ADR-0005)은 잠정값이다. G4 비교 전에 확정해야 한다.
2. CFD를 v2606으로 재현할 환경(네트워크 허용 또는 다른 머신)이 있는지.
3. 저장소를 사용자 PC로 옮기는 방법(비공개 GitHub 저장소 또는 파일 복사). 옮긴 뒤 WSL v2606 CFD 재현과 Isaac 스모크를 Remote Control로 실행.

## 재현 명령
```bash
python -m pip install -e ".[test]"
python -m rocket_dt validate-physics
python -m rocket_dt experiments --n-suite 30
python scripts/run_cfd_batch.py --engines conical_A bell_B --levels 1 2 3 --jobs 4 --out runs/cfd
python scripts/run_cfd_batch.py --post-only --out runs/cfd   # QA/GCI/맵만 다시 계산
python scripts/run_cfd_experiments.py          # CFD 맵 vs 1D 맵 비행 비교
python -m pip install -e ".[video]" && python scripts/render_replay.py   # runs/video/*.mp4 (cloud-v1912 리플레이, Isaac 아님)
```
