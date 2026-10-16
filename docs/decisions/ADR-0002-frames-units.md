# ADR-0002 좌표계·단위 규약

- 상태: 채택. 구현: `src/rocket_dt/dynamics/frames.py` (유일한 규약 정의 위치)

- SI 단위만 사용. 스키마 로더는 `units: SI`가 아니면 거부한다.
- World: 오른손, Z-up, 원점은 착륙 패드 표면.
- Body: 오른손, +Z = 기수 방향, 원점 = 기체 길이 중앙의 축선. CoM은 원점이 아니며 질량 모델이 계산한다.
- 메인 엔진 추력은 +Z_body 방향, 짐벌 피벗에 작용. 짐벌 방향 벡터 d = R_y(δy) R_x(δx) ẑ.
- 쿼터니언: (w, x, y, z), Hamilton 곱, v_world = R(q) v_body. Isaac Lab root state와 같은 순서.
- 참조 적분기의 병진 상태는 **CoM** 위치·속도(world). Isaac root pose는 링크 원점이므로 비교 시 p_origin = p_com − R·com_body로 변환한다.
- 팽창비 ε = Ae/At (면적비). 테스트 `test_contour_area_ratio_and_lengths`가 반경비와 혼동하지 않음을 확인.
- g0 = 9.80665 m/s² (NASA Glenn 페이지는 9.8 표기; Isp 정의용 상수로 고정).
