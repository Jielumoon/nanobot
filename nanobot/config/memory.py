"""Pure configuration resolution shared by memory runtime and settings saves."""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from nanobot.config.loader import resolve_config_env_vars
from nanobot.config.network import NetworkConfig
from nanobot.config.schema import MemoryEmbeddingConfig, ProviderConfig, ProvidersConfig
from nanobot.providers.registry import find_by_name


@dataclass(frozen=True)
class EmbeddingProvider:
    provider: ProviderConfig
    api_base: str
    proxy: str | None


def resolve_embedding_provider(
    config: MemoryEmbeddingConfig, providers: ProvidersConfig, network: NetworkConfig,
) -> EmbeddingProvider:
    provider = getattr(providers, config.provider, None)
    if not isinstance(provider, ProviderConfig):
        raise ValueError(f"记忆向量配置无效：providers.{config.provider} 不存在")
    provider = resolve_config_env_vars(provider)
    if not provider.api_key or not provider.api_key.strip():
        raise ValueError(f"记忆向量配置无效：请设置 providers.{config.provider}.apiKey")
    spec = find_by_name(config.provider)
    base = provider.api_base or (spec.default_api_base if spec else None)
    if not base and config.provider == "openai":
        base = "https://api.openai.com/v1"
    if not base:
        raise ValueError(f"记忆向量配置无效：请设置 providers.{config.provider}.apiBase")
    try:
        parsed = urlsplit(base.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port == 0:
            raise ValueError("invalid endpoint")
    except ValueError:
        raise ValueError(
            f"记忆向量配置无效：providers.{config.provider}.apiBase 须为 HTTP(S) 地址",
        ) from None
    base = urlunsplit(parsed._replace(path=parsed.path.rstrip("/")))
    return EmbeddingProvider(provider, base, network.resolve_proxy(provider.use_proxy))
