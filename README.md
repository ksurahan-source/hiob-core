# hiob-core — 공유 런타임 (추출 매니페스트, Phase 0.2)

전 행성이 depend하는 공유 런타임. **확실한 길**: 14 importer를 깨는 물리 이동은 Modal import 테스트로 검증해야 하므로, 이 매니페스트로 정확히 grounding한 뒤 단계별 검증 이동한다.

## 추출 대상 공개 표면 (ground-truth grep)

### `llm_runtime` (628줄, **14 importers**)
`load_prompt` · `load_localized_prompt` · `resolve_model` · `estimate_cost_cents` · `llm_json` · `llm_vision_json` · `llm_json_cached` · `langfuse_log` · `JsonRepairError`
(`_`접두 = 내부, 비공개)

### `model_providers` (203줄)
`normalize_script_model` · `resolve_script_model` · `script_model_id` · `normalize_interpret_model` · `resolve_interpret_model` · `interpret_model_id` · `normalize_asset_engine` · `resolve_asset_engine` · `engine_is_live` · `engine_produces_video`

### `hiob_platform` (인프라 모듈만)
`client` · `storage` · `runs` · `role_artifacts` · `notify` · `placement` · `brand_kit` · `pronunciation`
> ⚠️ **`team.py`는 제외** — 조립 로직 = **hiob-atropos 소유**(공유 런타임 아님). 매니페스트 정정.

## 안전 이동 시퀀스 (각 단계 검증)
1. 모듈을 `hiob_core/`로 **복사**(이동 아님) → `hiob_core/__init__.py`가 공개 표면 re-export.
2. `pip install -e packages/hiob-core` → Modal Image에 `add_local_python_source` 또는 패키지 depend.
3. **14 importer 재작성**: `from .llm_runtime import X` → `from hiob_core import X` (한 번에 하나씩).
4. **검증 게이트**: `python -m py_compile` + Modal 함수 import smoke(`modal run ... --help` 또는 로컬 import 테스트) 통과.
5. 전 importer 전환 확인 후 원본 삭제(drift 방지).

## 의존 방향 (불변)
hiob-core는 **아무 행성도 import 안 함**(최하층). 행성·hiob-data·hiob-contracts가 hiob-core를 depend. 순환 금지.

## 상태
Phase 0.2 — 매니페스트 + 스캐폴드 완료(이 문서 + pyproject). 다음 = 복사+re-export+importer 재작성(Modal import 검증 동반). 그 뒤 0.3 hiob-data governor.
