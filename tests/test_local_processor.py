"""Tests for LocalProcessor helper methods (no Ollama required)."""

import json
import pytest
from latent_gate.config import PipelineConfig
from latent_gate.local_processor import LocalProcessor


class TestLocalProcessorHelpers:

    def setup_method(self):
        config = PipelineConfig(enable_caching=False)
        self.processor = LocalProcessor(config)

    def test_clean_json_with_markdown_fences(self):
        raw = """```json
{"scene_type": "indoor", "objects": ["table"]}
```"""
        cleaned = self.processor._clean_json_output(raw)
        data = json.loads(cleaned)
        assert data["scene_type"] == "indoor"

    def test_clean_json_already_clean(self):
        raw = '{"scene_type": "outdoor"}'
        cleaned = self.processor._clean_json_output(raw)
        data = json.loads(cleaned)
        assert data["scene_type"] == "outdoor"

    def test_clean_json_with_whitespace(self):
        raw = '   \n  {"key": "value"}  \n  '
        cleaned = self.processor._clean_json_output(raw)
        data = json.loads(cleaned)
        assert data["key"] == "value"

    def test_parse_restructured_output(self):
        text = """SCENE: indoor
DESC: A kitchen with appliances
OBJECTS: stove, fridge, table
ACTIONS: cooking
TEXT: none"""
        payload = self.processor._parse_restructured_output(text)
        assert payload.scene_type == "indoor"
        assert "kitchen" in payload.scene_description
        assert "stove" in payload.objects_detected
        assert "cooking" in payload.actions_activities

    def test_parse_restructured_empty(self):
        payload = self.processor._parse_restructured_output("")
        assert payload.scene_type == ""
        assert payload.objects_detected == []

    def test_encode_image_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.processor.encode_image_to_base64("nonexistent_image.jpg")

    def test_encode_image_bad_format(self):
        # Create a temp file with bad extension
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"fake data")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                self.processor.encode_image_to_base64(temp_path)
        finally:
            os.unlink(temp_path)
