"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root (directory containing config/)
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return current.parent.parent.parent


def load_env() -> None:
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    load_dotenv(env_path)


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get an environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raise error if not set

    Returns:
        The environment variable value

    Raises:
        ValueError: If required is True and the variable is not set
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable not set: {key}")
    return value or ""


class AppConfig:
    """Application configuration container."""

    def __init__(self):
        """Initialize configuration from files and environment."""
        load_env()

        self.project_root = get_project_root()

        self.eodhd_api_key = get_env("EODHD_API_KEY", required=True)
        self.sec_contact_email = get_env("SEC_CONTACT_EMAIL", required=True)

        self.mysql_host = get_env("MYSQL_HOST", "localhost")
        self.mysql_port = int(get_env("MYSQL_PORT", "3306"))
        self.mysql_user = get_env("MYSQL_USER", "biopharma")
        self.mysql_password = get_env("MYSQL_PASSWORD", "")
        self.mysql_database = get_env("MYSQL_DATABASE", "biopharma_monitor")

        self.slack_webhook_url = get_env("SLACK_WEBHOOK_URL")
        self.discord_webhook_url = get_env("DISCORD_WEBHOOK_URL")

        filters_path = self.project_root / "config" / "filters.yaml"
        self.filters = load_config(str(filters_path)) if filters_path.exists() else {}

        eodhd_path = self.project_root / "config" / "eodhd.yaml"
        self.eodhd_config = load_config(str(eodhd_path)) if eodhd_path.exists() else {}

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.project_root / "data"

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.project_root / "logs"
