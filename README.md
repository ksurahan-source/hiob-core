# hiob-core (발판)
모든 행성이 쓰는 **공유 런타임**: `llm_runtime`(LLM 호출/재시도/비용가드)·`model_providers`(모델 선택)·플랫폼 유틸.
- 의존: hiob-contracts. (platform-py 흡수 예정)
- 자립: `pytest` → 19 passed.
