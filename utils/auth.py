"""
Authentication utilities for Hugging Face Hub integration.

This module provides secure and centralized authentication management for HF Hub access.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import huggingface_hub
from huggingface_hub import HfApi


class HFAuthError(RuntimeError):
    """Raised when authentication fails."""
    pass


class HFScopesError(PermissionError):
    """Raised when token scopes are insufficient."""
    pass


def get_hf_token() -> Optional[str]:
    """
    Get HF token from various sources in priority order.

    Returns:
        Token string if found, None otherwise. Never logs the token value.
    """
    # Priority 1: Environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        logging.debug("HF token found in environment variable")
        return token

    # Priority 2: config/tokens.json
    tokens_file = Path("config/tokens.json")
    if tokens_file.exists():
        try:
            with open(tokens_file, 'r', encoding='utf-8') as f:
                tokens_data = json.load(f)

            # Try both possible key names for backwards compatibility
            token = tokens_data.get("huggingface", {}).get("token") or tokens_data.get("huggingface_token")
            if token:
                logging.debug("HF token found in config/tokens.json")
                return token
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logging.warning(f"Failed to read config/tokens.json: {e}")

    # Priority 3: HF CLI cache
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # If whoami succeeds, we have a cached token
            logging.debug("HF token found in CLI cache")
            # We don't extract the actual token, just verify it exists
            # The HF library will use it automatically
            return "cli_cached_token"  # Placeholder to indicate token exists
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        logging.debug("HF CLI not available or no cached token")

    logging.debug("No HF token found in any source")
    return None


def _check_file_permissions(file_path: Path) -> None:
    """Check file permissions and warn if too permissive."""
    if not file_path.exists():
        return

    try:
        # Get file permissions (Unix-style)
        stat_info = file_path.stat()
        permissions = oct(stat_info.st_mode)[-3:]  # Last 3 digits

        # Warn if permissions are more permissive than 600 (rw-------)
        if int(permissions) > 600:
            logging.warning(
                f"Token file {file_path} has permissive permissions ({permissions}). "
                "Consider restricting to 600 (rw-------) for security."
            )
    except OSError as e:
        logging.debug(f"Could not check permissions for {file_path}: {e}")


def ensure_hf_login(required_scopes: Tuple[str, ...] = ("read",)) -> str:
    """
    Ensure HF authentication and validate scopes.

    Args:
        required_scopes: Required token scopes (default: ("read",))

    Returns:
        The validated token string

    Raises:
        HFAuthError: If authentication fails
        HFScopesError: If token scopes are insufficient
    """
    # Get token
    token = get_hf_token()
    if not token:
        raise HFAuthError(
            "No Hugging Face token found. Please set HF_TOKEN environment variable, "
            "add token to config/tokens.json, or run 'huggingface-cli login'"
        )

    # Check file permissions if token came from config file
    tokens_file = Path("config/tokens.json")
    if tokens_file.exists() and not os.getenv("HF_TOKEN"):
        _check_file_permissions(tokens_file)

    # Authenticate (skip if using CLI cached token)
    if token != "cli_cached_token":
        try:
            huggingface_hub.login(token=token, add_to_git_credential=False)
        except Exception as e:
            raise HFAuthError(f"Failed to authenticate with Hugging Face: {e}")

    # Verify connection and check scopes
    try:
        whoami_info = huggingface_hub.whoami()

        # Extract actual token scopes from auth info
        auth_info = whoami_info.get("auth", {})
        access_token_info = auth_info.get("accessToken", {})
        token_role = access_token_info.get("role", "read")  # Default to read

        # Handle different scope formats
        if isinstance(token_role, str):
            actual_scopes = [token_role]
        elif isinstance(token_role, list):
            actual_scopes = token_role
        else:
            # Fallback: assume basic read access
            actual_scopes = ["read"]

        # Check if all required scopes are present
        missing_scopes = [scope for scope in required_scopes if scope not in actual_scopes]
        if missing_scopes:
            raise HFScopesError(
                f"Token missing required scopes: {missing_scopes}. "
                f"Token has: {actual_scopes}. "
                "Please generate a new token with appropriate permissions."
            )

        logging.info(f"Successfully authenticated with Hugging Face as {whoami_info.get('name', 'unknown')}")
        return token if token != "cli_cached_token" else "authenticated_via_cli"

    except Exception as e:
        if isinstance(e, HFScopesError):
            raise
        raise HFAuthError(f"Failed to verify Hugging Face authentication: {e}")


def get_hf_api_client(required_scopes: Tuple[str, ...] = ("read",)) -> HfApi:
    """
    Get authenticated Hugging Face API client.

    Args:
        required_scopes: Required token scopes (default: ("read",))

    Returns:
        Authenticated HfApi client instance

    Raises:
        HFAuthError: If authentication fails
        HFScopesError: If token scopes are insufficient
    """
    # Ensure authentication
    ensure_hf_login(required_scopes)

    # Return authenticated API client
    try:
        api = HfApi()
        # Verify the client works by making a simple API call
        api.whoami()
        return api
    except Exception as e:
        raise HFAuthError(f"Failed to create authenticated HF API client: {e}")