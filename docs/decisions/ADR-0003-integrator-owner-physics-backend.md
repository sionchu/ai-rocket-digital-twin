# ADR-0003 적분 소유자, physics backend, 추력 적용 방식

- 상태: 채택 (Isaac 부분은 BLOCKED로 미검증)

## 결정
1. 한 실행에는 적분기 하나. Isaac 실행에서는 PhysX가 권위 있는 적분기이고, Python RK4 참조 모델(`dynamics/reference_6dof.py`)은 Isaac이 없는 실행과 오프라인 한 스텝 비교에만 쓴다. RocketPy는 같은 시간축에서 돌리지 않는다.
2. Physics backend는 PhysX. Newton은 ThrusterCfg 미지원(3.0-EA 소스 주석)이라 보류.
3. 추력 적용: Propulsion Adapter가 추력기별 (F_i, r_i)를 body frame으로 내고, `isaac_env/wrench_bridge.py`가 ΣF와 Σ(r_i − r_CoM)×F_i를 직접 계산해 `permanent_wrench_composer.set_forces_and_torques(..., positions=None, is_global=False)`로 넘긴다. 2.3.x는 positions를 링크 원점 기준, 3.0은 CoM 기준으로 해석하므로 positions를 쓰지 않는 쪽이 버전에 독립적이다.
4. 매 `_apply_action()`마다 덮어쓴다(permanent composer는 reset 전까지 유지되므로 엔진 정지 시 0을 명시적으로 쓴다).

## 미검증 (G2 전에 확인)
- 2.3.2에서 positions=None일 때 힘이 실제로 CoM에 작용하는지: `scripts/isaac_force_smoke.py`(미작성/NOT_RUN)로 Python 참조와 한 스텝 비교.
- 추진제 소모에 따른 질량·CoM·관성 갱신 API와 갱신 빈도.
