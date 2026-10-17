# HANDOFF

갱신: 2026-10-17

## 지금 상태
- 단계 0(조사), 단계 1(P0 vertical slice 중 Python 부분), 단계 2(CFD) 일부, 단계 3(PD 기준선)을 진행했다. 게이트별 판정은 `PROJECT_LEDGER.md`.
- 결과는 실행 환경 라벨로 나눈다: `cloud-v1912`(클라우드, OpenFOAM v1912), `wsl-v2606`(사용자 PC WSL, OpenFOAM v2606), `win-isaac`(사용자 PC Windows, Isaac Lab 3.0.0 + Isaac Sim 6.0.1-rc.7).
- 2026-10-17 사용자 PC에서:
  - `wsl-v2606` CFD 6케이스 재현 PASS. v1912 대비 창 평균 추력·ṁ 상대차 < 1e-4. v2606 blockMesh는 `-merge-points`가 필요하다(버전 조건부로 `run_cfd_batch.py`에 반영). 결과 `runs/cfd_wsl-v2606/`.
  - `win-isaac` 단일 강체 힘 스모크 PASS (선 ≤ 0.26%, 각 ≤ 0.24%). 결과 `runs/isaac_smoke_win-isaac/`.
- 설치된 Isaac은 목표 고정(Isaac Lab 2.3.2 + Isaac Sim 5.1.0)이 아니라 3.0.0 + 6.0.1-rc.7이다. 3.x는 루트 쿼터니언이 xyzw다(rocket_dt는 wxyz).
- 알려진 함정: `set_inertias_index`로 축대칭 관성을 쓰면 PhysX CoM 프레임이 깨진다. 질량 특성은 USD MassAPI로 리셋 전에 지정한다.

## PPO (단계 4, 브랜치 `pc/ppo-v0`)
- `VtvlEnv`(Isaac Lab 3.0.0, ADR-0009)와 rsl_rl PPO 학습·평가·영상 스크립트가 있다. 결과 `runs/ppo_win-isaac/summary.md`.
- 최선 정책 r3 model_398: conical_A 시드 1000–1029에서 29/30, 보류 시드 2000–2029에서 28/30. PD 기준선은 30/30(다른 시뮬레이터).
- 약점: 착지 |vz| P95 1.97 m/s로 한계(2.0)에 붙어 있다. 재학습 없이 엔진을 바꾸면 bell_B 20/30, bell_C 5/30.

## 다음 액션 하나
같은 Isaac env 안에서 PD 기준선을 torch로 돌려 시뮬레이터 차이 없이 PPO와 비교한다. 이어서 착지 속도 여유(예: |vz| 목표 1.0–1.5 m/s)를 보상에 넣고, 엔진·맵을 섞어 학습(A/B/C 무작위)해서 E3 강건성을 다시 잰다.

## 열린 질문 (사용자 결정 필요)
1. 착륙 성공 임계값(ADR-0005)은 잠정값이다. G4 비교 전에 확정해야 한다.
2. ADR-0009 고정 질량(착지 때 7.1% 무거움)을 v0에서 받아들일지, 질량 갱신 방법을 더 찾을지.
3. Isaac 버전 고정: 설치된 3.0.0 + 6.0.1-rc.7을 새 고정값으로 할지, 2.3.2 + 5.1.0을 따로 설치할지 (ADR-0001 갱신 필요).

## 재현 명령
```bash
python -m pip install -e ".[test]"
python -m rocket_dt validate-physics
python -m rocket_dt experiments --n-suite 30
python scripts/run_cfd_batch.py --engines conical_A bell_B --levels 1 2 3 --jobs 4 --out runs/cfd
python scripts/run_cfd_batch.py --post-only --out runs/cfd   # QA/GCI/맵만 다시 계산
# wsl-v2606 (WSL 내부 클론에서): export FOAM_BASHRC=/usr/lib/openfoam/openfoam2606/etc/bashrc
python3 scripts/run_cfd_batch.py --engines conical_A bell_B --levels 1 2 3 --jobs 6 --out runs/cfd_wsl-v2606
# win-isaac (Windows):
# C:\dev\IsaacLab\isaaclab.bat -p scripts\isaac_force_smoke.py --headless --out runs\isaac_smoke_win-isaac
# isaaclab.bat -p scripts\ppo_train.py --run_name r0_base --num_envs 4096 --max_iterations 150
# isaaclab.bat -p scripts\ppo_train.py --run_name r3_guided --max_iterations 250 --shaping guided --phi_pos_m 2 --phi_tilt_rad 0.3 --rew_td_success 20 --resume_from runs\ppo_win-isaac\train\r0_base\model_149.pt
# isaaclab.bat -p scripts\ppo_eval.py --checkpoint runs\ppo_win-isaac\train\r3_guided\model_398.pt [--engine bell_B] [--seed0 2000]
# isaaclab.bat -p scripts\ppo_eval.py --checkpoint ... --video --video_episodes 3 --video_name late --enable_cameras
python scripts/run_cfd_experiments.py          # CFD 맵 vs 1D 맵 비행 비교
```
