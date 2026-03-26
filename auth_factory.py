"""
Authentication factory for datacloud-mcp-query.

Provides auto-detection logic to choose the appropriate authentication method.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auth_interface import AuthProvider

from sf_cli_auth import SFCLIAuth
from oauth import OAuthConfig, OAuthConfigError, OAuthSession

# Get logger for this module
logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Exception raised when authentication cannot be configured."""
    pass


def _build_error_message() -> str:
    """
    Build a comprehensive error message based on what's configured.

    Returns:
        str: Detailed error message with setup instructions
    """
    sf_org_alias = os.getenv("SF_ORG_ALIAS")
    sf_client_id = os.getenv("SF_CLIENT_ID")
    sf_client_secret = os.getenv("SF_CLIENT_SECRET")

    base_message = "No authentication method is configured.\n\n"

    # Determine which specific error scenario we're in
    if sf_org_alias and not sf_client_id:
        return base_message + (
            "SF_ORG_ALIAS is set, but SF CLI authentication failed.\n"
            "Ensure you've authenticated via:\n"
            f"  sf org login web --alias {sf_org_alias}\n\n"
            "Alternatively, configure OAuth authentication with:\n"
            "  export SF_CLIENT_ID='your_client_id'\n"
            "  export SF_CLIENT_SECRET='your_client_secret'\n"
            "See CONNECTED_APP_SETUP.md for OAuth setup instructions.\n"
        )

    if sf_client_id and not sf_client_secret:
        return base_message + (
            "SF_CLIENT_ID is set but SF_CLIENT_SECRET is missing.\n"
            "For OAuth authentication, both are required:\n"
            "  export SF_CLIENT_SECRET='your_client_secret'\n"
        )

    if sf_client_secret and not sf_client_id:
        return base_message + (
            "SF_CLIENT_SECRET is set but SF_CLIENT_ID is missing.\n"
            "For OAuth authentication, both are required:\n"
            "  export SF_CLIENT_ID='your_client_id'\n"
        )

    # No configuration at all - provide full setup instructions
    return base_message + (
        "Choose one of the following authentication methods:\n\n"
        "Method 1: SF CLI (Recommended)\n"
        "  1. Install SF CLI: https://developer.salesforce.com/tools/salesforcecli\n"
        "  2. Authenticate: sf org login web --alias myorg\n"
        "  3. Set environment variable: export SF_ORG_ALIAS='myorg'\n\n"
        "Method 2: OAuth PKCE\n"
        "  1. Create a Connected App (see CONNECTED_APP_SETUP.md)\n"
        "  2. Set environment variables:\n"
        "     export SF_CLIENT_ID='your_client_id'\n"
        "     export SF_CLIENT_SECRET='your_client_secret'\n"
    )


def create_auth_provider() -> AuthProvider:
    """
    Create an authentication provider with auto-detection.

    Auto-detection logic:
    1. If SF_ORG_ALIAS is set and SF CLI is available:
       - Try SF CLI authentication
       - If successful, return SFCLIAuth instance
       - If failed, log warning and continue to OAuth fallback
    2. If SF_CLIENT_ID and SF_CLIENT_SECRET are set:
       - Return OAuthSession instance
    3. Otherwise:
       - Raise AuthenticationError with setup instructions

    Returns:
        AuthProvider: An initialized authentication provider

    Raises:
        AuthenticationError: If no authentication method can be configured
    """
    sf_org_alias = os.getenv("SF_ORG_ALIAS")

    # Try SF CLI authentication first if configured
    if sf_org_alias and SFCLIAuth.is_available():
        logger.info(f"SF_ORG_ALIAS is set to '{sf_org_alias}', attempting SF CLI authentication")
        try:
            auth_provider = SFCLIAuth(sf_org_alias)
            # Test authentication by calling get_token()
            auth_provider.get_token()
            logger.info(f"Successfully configured SF CLI authentication with org: {sf_org_alias}")
            return auth_provider
        except Exception as e:
            logger.warning(
                f"SF CLI authentication failed: {str(e)}\n"
                "Falling back to OAuth if configured"
            )

    # Try OAuth authentication if configured
    if OAuthSession.is_available():
        logger.info("OAuth credentials configured, using OAuth PKCE authentication")
        try:
            oauth_session = OAuthSession(OAuthConfig.from_env())
            logger.info("Successfully configured OAuth PKCE authentication")
            return oauth_session
        except OAuthConfigError as e:
            logger.warning(f"OAuth configuration failed: {e}")


    # No authentication method available
    error_message = _build_error_message()
    logger.error(f"Authentication configuration failed:\n{error_message}")
    raise AuthenticationError(error_message)
