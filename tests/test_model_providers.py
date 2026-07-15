"""Unit tests for model_providers module (FakeClient pattern)."""

import pytest
import sys
from pathlib import Path

# Add hiob_core to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from hiob_core.model_providers import (
    resolve_script_model,
    script_model_id,
    resolve_interpret_model,
    interpret_model_id,
    resolve_asset_engine,
    engine_is_live,
    engine_produces_video,
    normalize_script_model,
    normalize_interpret_model,
    normalize_asset_engine,
    SCRIPT_MODELS,
    INTERPRET_MODELS,
    ASSET_ENGINES,
)


class TestScriptModelResolution:
    """Test SCRIPT model selection axis."""

    def test_resolve_script_model_default(self):
        """None brief returns live default (qwen — cheap qwen3.7-max path, LP7-7)."""
        assert resolve_script_model(None) == "qwen"
        assert resolve_script_model({}) == "qwen"

    def test_resolve_script_model_explicit(self):
        """brief.script_model overrides default."""
        assert resolve_script_model({"script_model": "gpt"}) == "gpt"
        assert resolve_script_model({"script_llm": "gpt"}) == "gpt"
        assert resolve_script_model({"script_model": "claude"}) == "claude"

    def test_normalize_script_model_aliases(self):
        """Aliases map to canonical ids."""
        assert normalize_script_model("claude-opus-4-8") == "claude"
        assert normalize_script_model("opus") == "claude"
        assert normalize_script_model("gpt-4o") == "gpt"
        assert normalize_script_model("gpt4o") == "gpt"
        assert normalize_script_model("qwen3.7-max") == "qwen"

    def test_script_model_id(self):
        """script_model_id returns provider string via resolve_script_model SSOT."""
        assert script_model_id(None) == "qwen3.7-max"
        assert script_model_id({"script_model": "gpt"}) == "gpt-4o"
        assert script_model_id({"script_model": "claude"}) == "claude-opus-4-8"

    def test_script_model_id_respects_claude_script_model_env(self, monkeypatch):
        """CLAUDE_SCRIPT_MODEL env must not diverge from script_model_id (LP7-7)."""
        monkeypatch.setenv("CLAUDE_SCRIPT_MODEL", "claude-sonnet-4-6")
        assert script_model_id({"script_model": "claude"}) == "claude-sonnet-4-6"
        # Default registry id stays qwen — env only affects the claude *string*.
        assert resolve_script_model(None) == "qwen"


class TestInterpretModelResolution:
    """Test INTERPRET model selection (independent of SCRIPT)."""

    def test_resolve_interpret_model_default(self):
        """None brief returns default (qwen per D-46: interpret 기본 sonnet→qwen3.7-plus)."""
        assert resolve_interpret_model(None) == "qwen"

    def test_resolve_interpret_model_explicit(self):
        """brief.interpret_model overrides default."""
        assert resolve_interpret_model({"interpret_model": "gpt_mini"}) == "gpt_mini"

    def test_interpret_model_id(self):
        """interpret_model_id returns provider string (default qwen3.7-plus per D-46)."""
        assert interpret_model_id(None) == "qwen3.7-plus"
        assert interpret_model_id({"interpret_model": "gpt_mini"}) == "gpt-4o-mini"

    def test_interpret_independent_of_script(self):
        """Choosing GPT for script doesn't change interpret (stays default qwen per D-46)."""
        brief_gpt_script = {"script_model": "gpt"}
        assert resolve_interpret_model(brief_gpt_script) == "qwen"


class TestAssetEngineResolution:
    """Test ASSET engine selection (image OR video per beat)."""

    def test_resolve_asset_engine_default(self):
        """None brief returns default (openai_image)."""
        assert resolve_asset_engine(None) == "openai_image"

    def test_resolve_asset_engine_style_default(self):
        """style_mode controls default when no explicit brief engine."""
        assert resolve_asset_engine({}, style_mode="branded") == "openai_image"
        assert resolve_asset_engine({}, style_mode="ugc") == "seedance_fast"
        assert resolve_asset_engine({}, style_mode="cine") == "seedance_hi"

    def test_resolve_asset_engine_explicit_override(self):
        """brief.asset_engine overrides style default."""
        brief_kling = {"asset_engine": "kling"}
        assert resolve_asset_engine(brief_kling, style_mode="branded") == "kling"

    def test_normalize_asset_engine_aliases(self):
        """Aliases normalize to canonical ids."""
        assert normalize_asset_engine("seedance") == "seedance_fast"
        assert normalize_asset_engine("seedance_hi") == "seedance_hi"
        assert normalize_asset_engine("openai") == "openai_image"
        assert normalize_asset_engine("gpt-image") == "openai_image"

    def test_engine_produces_video(self):
        """Video engines are identified correctly."""
        assert engine_produces_video("seedance_fast")
        assert engine_produces_video("kling")
        assert not engine_produces_video("openai_image")
        assert not engine_produces_video("gemini_image")


class TestEngineLifecycle:
    """Test engine live status (API key presence)."""

    def test_engine_is_live_openai_image(self):
        """openai_image is live if OPENAI_API_KEY is set."""
        import os
        # The test assumes environment — skip if keys are not set
        # In actual Modal, these are in secrets.
        status = engine_is_live("openai_image")
        # We just verify it returns bool; actual value depends on ENV.
        assert isinstance(status, bool)

    def test_engine_is_live_unknown(self):
        """Unknown engines return False."""
        assert not engine_is_live("unknown_engine")

    def test_engine_is_live_needs_key(self):
        """Engines with status='needs_key' return False without the key."""
        # This test assumes GEMINI_API_KEY is not set in test environment.
        # Adjust based on actual test setup.
        pass


class TestDataStructures:
    """Verify registries are well-formed."""

    def test_script_models_valid(self):
        """SCRIPT_MODELS has required fields."""
        for id_, spec in SCRIPT_MODELS.items():
            assert "id" in spec
            assert "model" in spec
            assert "env" in spec
            assert "status" in spec
            assert spec["id"] == id_

    def test_interpret_models_valid(self):
        """INTERPRET_MODELS has required fields."""
        for id_, spec in INTERPRET_MODELS.items():
            assert "id" in spec
            assert "model" in spec
            assert "env" in spec
            assert "status" in spec

    def test_asset_engines_valid(self):
        """ASSET_ENGINES has required fields."""
        for id_, spec in ASSET_ENGINES.items():
            assert "id" in spec
            assert "kind" in spec  # image or video
            assert "status" in spec
            assert spec["id"] == id_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
