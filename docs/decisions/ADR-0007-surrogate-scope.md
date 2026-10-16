# ADR-0007 Surrogate 사용 범위

- 상태: 채택. 실행 상태: NOT_RUN

1. 제어 정책 학습·평가와 E1–E4 비교는 CFD QA를 통과한 맵(또는 명시적으로 표기한 1D 맵)으로만 한다. Surrogate 예측을 학습 환경의 진실값으로 쓰지 않는다 (설계안 6장).
2. 첫 surrogate는 GP/작은 MLP로 설계·작동 feature(At, ε, 윤곽 각도, 길이 비, Pc, Pa) → S, ṁ를 예측한다. design-ID를 입력으로 쓰지 않는다.
3. 데이터 분할은 설계 단위. 현재 CFD 설계는 2개(A, B)뿐이라 held-out 평가가 불가능하다. 최소 8–12개 설계의 CFD가 쌓이기 전에는 surrogate를 학습하지 않는다.
4. 공개 데이터(`taarg/MultiFidelity_Nozzle_JetFlows` 등)는 로켓 노즐 Isp/추력 라벨이 없어 파이프라인 점검용으로만 고려한다.
