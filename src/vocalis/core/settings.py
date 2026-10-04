"""Runtime configuration, read from the environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cerebras_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = None
    deepgram_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_project_name: str = "vocalis"

    vocalis_talker_models: str = Field(
        default="cerebras/qwen-3.8-27b,cerebras/gpt-oss-120b,groq/openai/gpt-oss-120b"
    )
    vocalis_planner_models: str = Field(default="gemini/gemini-flash-latest,cerebras/gpt-oss-120b")
    vocalis_judge_models: str = Field(default="gemini/gemini-flash-lite-latest,groq/openai/gpt-oss-120b")
    vocalis_rep_models: str = Field(default="ollama_chat/qwen3:8b,groq/openai/gpt-oss-20b")
    vocalis_vision_model: str = "gemini-flash-latest"

    @property
    def talker_models(self) -> list[str]:
        return _csv(self.vocalis_talker_models)

    @property
    def planner_models(self) -> list[str]:
        return _csv(self.vocalis_planner_models)

    @property
    def judge_models(self) -> list[str]:
        return _csv(self.vocalis_judge_models)

    @property
    def rep_models(self) -> list[str]:
        return _csv(self.vocalis_rep_models)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
