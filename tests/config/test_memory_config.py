import json

import pytest
from pydantic import ValidationError

from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config, MemoryConfig, MemoryEmbeddingConfig


def test_memory_defaults_and_camel_case_roundtrip(tmp_path):
    config = Config()
    memory = config.agents.defaults.memory
    assert memory.claims_enabled and memory.daily_notes and memory.people.enabled
    assert memory.inject_max_chars == 6000
    assert memory.embedding.model_dump(by_alias=True) == {
        "enabled": False, "provider": "openai", "model": "text-embedding-3-small",
        "dimensions": None, "batchSize": 32,
    }
    path = tmp_path / "config.json"
    save_config(config, path)
    saved = json.loads(path.read_text())["agents"]["defaults"]["memory"]
    assert saved["claimsEnabled"] and saved["dailyNotes"]
    assert saved["injectMaxChars"] == 6000
    assert load_config(path).agents.defaults.memory == memory


def test_memory_explicit_disable_and_embedding_settings_survive_roundtrip(tmp_path):
    config = Config.model_validate({"agents": {"defaults": {"memory": {
        "claimsEnabled": False, "dailyNotes": False, "people": {"enabled": False},
        "injectMaxChars": 0, "embedding": {
            "enabled": True, "provider": "private", "model": "text-v1",
            "dimensions": 512, "batchSize": 16,
        },
    }}}})
    path = tmp_path / "config.json"
    save_config(config, path)
    memory = load_config(path).agents.defaults.memory
    assert not memory.claims_enabled and not memory.daily_notes and not memory.people.enabled
    assert memory.embedding == MemoryEmbeddingConfig(
        enabled=True, provider="private", model="text-v1", dimensions=512, batch_size=16,
    )
    assert MemoryConfig.model_validate(memory.model_dump(by_alias=True)) == memory


@pytest.mark.parametrize("values", [
    {"dimensions": 0}, {"batchSize": 0}, {"batchSize": 257}, {"provider": ""}, {"model": ""},
])
def test_memory_embedding_rejects_invalid_configuration(values):
    with pytest.raises(ValidationError):
        MemoryEmbeddingConfig.model_validate(values)
