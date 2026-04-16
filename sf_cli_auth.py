"""
SF CLI authentication provider for datacloud-mcp-query.

Uses the Salesforce CLI (sf command) to obtain access tokens from an authenticated org.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Optional

# Get logger for this module
logger = logging.getLogger(__name__)

# Environment variable names
ENV_SF_ORG_ALIAS = "SF_ORG_ALIAS"


class SFCLIAuth:
    """
    Authentication provider that uses SF CLI (sf command) for token management.

    Caches tokens for 5 minutes to reduce subprocess overhead while maintaining freshness.
    """

    # Token cache duration (5 minutes)
    CACHE_DURATION_MINUTES = 5

    def __init__(self, org_alias: str):
        """
        Initialize SF CLI authentication with the specified org alias.

        Credentials are fetched lazily on first use, not during construction,
        so the MCP server can start quickly without blocking on subprocess calls
        and also so that credentials are not unncessarily fetched until needed.

        Args:
            org_alias: The SF CLI org alias or username

        Raises:
            Exception: If SF CLI binary is not installed
        """
        self.org_alias = org_alias
        self._token: Optional[str] = None
        self._instance_url: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(
            f"Initializing SF CLI authentication with org: {org_alias}")

        if not shutil.which("sf"):
            raise Exception(
                "SF CLI (sf command) not found in PATH. "
                "Install from: https://developer.salesforce.com/tools/salesforcecli")

    @staticmethod
    def is_configured() -> bool:
        """
        Check if SF CLI authentication is configured.

        Returns:
            bool: True if SF_ORG_ALIAS is set and sf command exists
        """
        has_alias = bool(os.getenv(ENV_SF_ORG_ALIAS))
        return has_alias and shutil.which("sf") is not None

    def _refresh_credentials(self) -> None:
        """
        Refresh credentials by calling SF CLI.

        Raises:
            Exception: If SF CLI command fails or returns invalid data
        """
        logger.info(f"Refreshing SF CLI credentials for org: {self.org_alias}")

        try:
            # Call SF CLI to get org details with JSON output
            result = subprocess.run(
                ["sf", "org", "display", "--target-org", self.org_alias, "--json"],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                check=True,
            )

            logger.debug(f"SF CLI command completed, parsing output")

            # Parse JSON output
            output = json.loads(result.stdout)

            # Extract access token and instance URL
            # SF CLI output format: {"result": {"accessToken": "...",
            # "instanceUrl": "..."}}
            result_data = output.get("result", {})
            access_token = result_data.get("accessToken")
            instance_url = result_data.get("instanceUrl")

            if not access_token or not instance_url:
                raise Exception(
                    f"SF CLI response missing required fields. "
                    f"Got accessToken={bool(access_token)}, instanceUrl={bool(instance_url)}")

            # Update cached values
            self._token = access_token
            self._instance_url = instance_url
            self._token_expiry = datetime.now() + timedelta(minutes=self.CACHE_DURATION_MINUTES)

            logger.info(
                f"Successfully refreshed SF CLI credentials, "
                f"cached until {self._token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")

        except subprocess.TimeoutExpired as e:
            raise Exception(
                f"SF CLI command timed out after 120 seconds") from e
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else e.stdout
            raise Exception(
                f"SF CLI command failed with exit code {e.returncode}. "
                f"Ensure you've authenticated via: sf org login web --alias {self.org_alias}\n"
                f"Error output: {error_output[:500]}") from e
        except json.JSONDecodeError as e:
            raise Exception(
                f"SF CLI returned invalid JSON. "
                f"Raw output (first 500 chars): {result.stdout[:500]}"
            ) from e
        except FileNotFoundError as e:
            raise Exception(
                "SF CLI (sf command) not found. "
                "Install from: https://developer.salesforce.com/tools/salesforcecli") from e

    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid cached token, refreshing if needed."""
        if not self._token_expiry or datetime.now() >= self._token_expiry:
            self._refresh_credentials()

    def get_token(self) -> str:
        """
        Get a valid access token from SF CLI.

        Returns:
            str: A valid Salesforce access token

        Raises:
            Exception: If SF CLI authentication fails
        """
        self._ensure_valid_token()
        return self._token

    def get_instance_url(self) -> str:
        """
        Get the Salesforce instance URL from SF CLI.

        Returns:
            str: The Salesforce instance URL

        Raises:
            Exception: If SF CLI authentication fails
        """
        self._ensure_valid_token()
        return self._instance_url
