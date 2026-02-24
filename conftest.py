"""
Unit test conftest — isolate API key environment variables so that
TestValidateAll tests are not affected by real keys in the developer's
or CI environment.
"""
import pytest

_API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BYTEZ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "SERPAPI_KEY",
]


@pytest.fixture(autouse=True)
def _clear_api_keys_from_env(monkeypatch):
    """Remove API key env vars for every unit test so Settings() behaves
    as if no keys are present unless the test explicitly provides them.
    Also disables .env file loading so local developer .env files don't
    leak real credentials into tests."""
    # Clear from os.environ
    for var in _API_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    # Disable .env file loading by patching Settings.model_config
    import config.settings as settings_module
    from pydantic_settings import SettingsConfigDict
    patched_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )
    monkeypatch.setattr(settings_module.Settings, "model_config", patched_config)