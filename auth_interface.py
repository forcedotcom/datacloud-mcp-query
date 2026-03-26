"""
Authentication protocol interface for datacloud-mcp-query.

Defines the common interface that all authentication providers must implement.
Uses Protocol for structural subtyping (duck typing).
"""
from typing import Protocol


class AuthProvider(Protocol):
    """
    Protocol defining the interface for authentication providers.

    All authentication implementations (OAuth, SF CLI, etc.) must provide these methods.
    """

    def get_token(self) -> str:
        """
        Get a valid access token for Salesforce API calls.

        Returns:
            str: A valid Salesforce access token

        Raises:
            Exception: If authentication fails or token cannot be obtained
        """
        ...

    def get_instance_url(self) -> str:
        """
        Get the Salesforce instance URL for API calls.

        Returns:
            str: The Salesforce instance URL (e.g., "https://example.my.salesforce.com")

        Raises:
            Exception: If instance URL cannot be determined
        """
        ...

    @staticmethod
    def is_configured() -> bool:
        """
        Check if the required environment variables for this authentication method are set.

        Returns:
            bool: True if the required configuration is present, False otherwise
        """
        ...
