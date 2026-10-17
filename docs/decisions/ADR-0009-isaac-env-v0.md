# ADR-0009 Isaac 착륙 env v0 (`win-isaac`): 고정 질량, PhysX 접촉, 보상

- 상태: 채택 (env v0 한정). 대상 설치본: Isaac Lab 3.0.0 (`v3.0.0-beta2.patch1-5-g418ca31b4`) + Isaac Sim 6.0.1-rc.7, PhysX, rsl-rl-lib 5.0.1
- 코드: `src/rocket_dt/isaac_env/rocket_env.py`, `torch_models.py`

## 1. 질량·CoM·관성은 에피소드 동안 고정
- 차량 USD(`build_vehicle_usd`)의 `UsdPhysics.MassAPI`에 추진제 가득(450 kg) 상태 값을 넣는다: m 1207.19 kg, CoM z −0.2276 m, I diag(4016.0, 4016.0, 320.06) kg·m².
- 추진제 질량은 torch로 적분해서 관측(`normalized_fuel`), 보상(연료), 지표(fuel_used)에만 쓴다. 강체 질량에는 반영하지 않는다.
- 이유 (둘 다 이 PC에서 실행해 확인):
  - `set_inertias_index`로 축대칭 관성(I_xx = I_yy)을 쓰면 PhysX `setCMassLocalPose pose is not valid` 오류가 나고 동역학이 틀어진다 (`scripts/isaac_force_smoke.py` 개발 중 관찰).
  - `set_masses_index`는 읽기값만 1127 kg으로 바뀌고 GPU 동역학에는 반영되지 않았다. 같은 힘에서 가벼운 env의 a_z가 무거운 env와 같았다(측정 6.7608, 기대 7.9367 m/s², 상대오차 14.8%). `scripts/ppo_env_smoke.py`의 `set_masses_probe`.
- 오차 크기: 착륙까지 약 80 kg을 쓰므로(PD 기준선 평균 78.8 kg) 착지 직전 실제 질량은 1127 kg이다. 고정 질량은 7.1% 무겁다. CoM은 실제로 0.099 m 아래(−0.327 m)로 내려가고, I_xx는 4.6%, I_zz는 3.9% 작아진다. 정책은 피드백으로 보정하지만 Python 참조와의 정량 비교에는 이 차이가 들어간다.
- 해제 조건: GPU 파이프라인에서 질량을 바꾸는 방법 확인(예: `set_masses_index` 뒤 시뮬레이션 뷰 재초기화, CPU 파이프라인, 또는 추진제를 별도 강체로 분리).

## 2. 접촉
- 다리 4개 끝에 반지름 0.15 m 구 충돌체(바닥점 = `gear.foot_positions_body_m`). PhysX 강체 접촉, 마찰 0.6, 반발 0.
- Python 참조의 다리별 스프링-댐퍼(k 1e5, c 8e3)와 다르다. 그래서 첫 접촉 상태(속도, 기울기, 위치 오차)는 충돌이 풀리기 **전** 상태로 기록한다: 물리 스텝마다 발 바닥 높이 ≤ 0.02 m가 되는 첫 순간에 기록하고 엔진을 끈다(simulate.py 규칙).

## 3. 에피소드와 판정
- 물리 200 Hz, 제어 50 Hz (decimation 4), 최대 60 s (`landing.yaml`).
- 리셋 분포는 `evaluation.experiments.randomized_scenarios`와 같다(고도 40–80 m, 수평 ±8 m, 속도, 기울기 ≤ 4°, 롤 각속도, 바람 ≤ 300 N). 학습은 torch 난수, 평가는 시드 1000–1029의 정확한 시나리오.
- 성공 판정은 `LANDING_LIMITS`(ADR-0005, 잠정) 그대로. 평가는 접지 후 3 s 정착 뒤 최종 기울기 ≤ 15°까지 본다. 학습은 접지 순간 종료(정착 생략)한다.
- 종료: 기울기 > 60°(전도), 고도 > 150 m 또는 수평 > 40 m(경계 이탈, 학습 보조), 60 s 시간 초과.
- 맵 invalid 셀을 쓴 스텝 수를 env별로 세서 `physics_valid`를 따로 보고한다.

## 4. 보상 (설계 선택, 결과를 보고 바꾸면 여기에 기록)
- 퍼텐셜 기반 셰이핑 r = 2·(Φ(s') − Φ(s)), Φ = −(|vz − vz_ref(h)|/2 + 수평오차/4 + |v_h|/1 + 기울기/0.1 + |ω|/0.5). vz_ref(h) = −clip(0.3 h, 0.8, 6)은 PD 기준선의 하강 프로파일(셰이핑 안내만, 정책이 따를 의무 없음).
- 연료 −0.004·스로틀/스텝, 행동 변화율 −0.01·|Δa|²/스텝.
- 접지 보상 5 − 2·(초과 |vz|) − 2·(초과 v_h) − 1·(초과 위치오차) − 20·(초과 기울기[rad]) − 0.5|vz| − 0.5·위치오차. 모든 한계를 만족하면 +10 추가.
- 전도·경계 이탈 −10, 시간 초과 −10.
- 퍼텐셜 셰이핑이라 "빨리 끝내기(추락)"로 셰이핑 손실을 피하는 유인이 없다(γ=0.995 근사).
