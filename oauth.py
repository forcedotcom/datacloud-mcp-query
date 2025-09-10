from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import sys
import base64
import hashlib
import secrets
import time
from threading import Thread
import http.server
import webbrowser
from urllib.parse import parse_qs, urlparse
from typing import Tuple

import requests
from rfc3986 import builder as uri_builder

# Get logger for this module
logger = logging.getLogger(__name__)


class OAuthConfig:
    def __init__(self, client_id: str, client_secret: str, login_root: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.login_root = login_root
        self.redirect_uri = redirect_uri

    @classmethod
    def from_env(cls) -> "OAuthConfig":
        client_id = os.getenv("SF_CLIENT_ID")
        client_secret = os.getenv("SF_CLIENT_SECRET")
        login_root = os.getenv("SF_LOGIN_URL", "login.salesforce.com")
        redirect_uri = os.getenv(
            "SF_CALLBACK_URL", "http://localhost:55556/Callback")

        missing = [name for name, val in {
            "SF_CLIENT_ID": client_id,
            "SF_CLIENT_SECRET": client_secret,
        }.items() if not val]
        if missing:
            print(
                f"Error: Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)

        return cls(client_id=client_id, client_secret=client_secret, login_root=login_root, redirect_uri=redirect_uri)


class _RequestHandler(http.server.BaseHTTPRequestHandler):  # pragma: no cover
    def do_GET(self):  # noqa: N802
        parts = urlparse(self.path)
        if parts.path.lower() != "/callback":
            self.send_error(404, "Not Found", "Not Found")
            return

        args = parse_qs(parts.query)
        self.server.oauth_result = args

        has_code = "code" in args
        response_content = f"Final Status: {has_code=}".encode("utf-8")
        response_content += b"\nYou can close this window now"
        self.send_response(200, "OK")
        self.send_header("Content-Type", "text")
        self.send_header("Content-Length", str(len(response_content)))
        self.end_headers()
        self.wfile.write(response_content)


def _generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge for OAuth flow"""
    code_verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(
            32)).decode("utf-8").rstrip("=")
    )

    challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = (
        base64.urlsafe_b64encode(challenge).decode("utf-8").rstrip("=")
    )

    return code_verifier, code_challenge


class OAuthSession:
    def __init__(self, config: OAuthConfig):
        self.config = config
        self.token: str | None = None
        self.exp: datetime | None = None
        self.instance_url: str | None = None

    def _run_oauth_flow(self, scopes: list[str]):
        logger.info(f"Starting OAuth flow with scopes: {scopes}")
        login_url = f"https://{self.config.login_root}/services/oauth2/authorize"
        token_exchange_url = f"https://{self.config.login_root}/services/oauth2/token"
        redirect_uri = self.config.redirect_uri

        code_verifier, code_challenge = _generate_pkce_pair()

        browser_uri: str = (
            uri_builder.URIBuilder(path=login_url)
            .add_query_from(
                {
                    "client_id": self.config.client_id,
                    "redirect_uri": redirect_uri,
                    "response_type": "code",
                    "scope": " ".join(scopes),
                    "prompt": "login",
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                }
            )
            .finalize()
            .unsplit()
        )

        parsed_redirect = urlparse(redirect_uri)
        port = parsed_redirect.port

        logger.debug(f"Starting OAuth callback server on localhost:{port}")
        server = http.server.HTTPServer(("localhost", port), _RequestHandler)
        server.allow_reuse_address = True
        t = Thread(target=server.handle_request, daemon=True)
        t.start()

        logger.info(f"Opening browser for OAuth authorization")
        logger.debug(f"Browser URI: {browser_uri}")
        webbrowser.open_new_tab(browser_uri)
        while t.is_alive():
            t.join(10)

        oauth_result_args = server.oauth_result

        if "code" not in oauth_result_args:
            error_msg = "OAuth authentication failed - no authorization code received"
            if "error" in oauth_result_args:
                error_msg += f". Error: {oauth_result_args['error'][0]}"
                if "error_description" in oauth_result_args:
                    error_msg += f" - {oauth_result_args['error_description'][0]}"
            raise Exception(error_msg)

        code = oauth_result_args["code"][0]
        logger.info(f"Authorization code received, exchanging for access token")

        response = requests.post(
            token_exchange_url,
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"Accept": "application/json"},
        )

        logger.info(f"Token exchange response: status={response.status_code}, elapsed={response.elapsed.total_seconds():.2f}s")

        if response.status_code >= 400:
            logger.error(f"Token exchange failed: {response.text}")

        response.raise_for_status()

        logger.info("Successfully obtained access token")
        return response.json()

    def ensure_access(self) -> str:
        if self.exp is not None and datetime.now() > self.exp:
            self.exp = None
            self.token = None

        if self.token is None:
            auth_info = self._run_oauth_flow(
                ["api", "cdp_query_api", "cdp_profile_api"])
            self.token = auth_info["access_token"]
            self.exp = datetime.now() + timedelta(minutes=110)
            self.instance_url = auth_info["instance_url"]

        return self.token

    def get_token(self) -> str:
        return self.ensure_access()

    def get_instance_url(self) -> str:
        self.ensure_access()
        return self.instance_url

    def create_dc_session(self) -> "OAuthDCSession":
        """Create an OAuthDCSession that manages a Data Cloud token via token exchange."""
        return OAuthDCSession(self)


class OAuthDCSession:
    def __init__(self, base_session: OAuthSession):
        self._base = base_session
        self.dc_token: str | None = None
        self.dc_exp: datetime | None = None
        self.dc_instance_url: str | None = None

    def _exchange_for_dc_token(self) -> dict:
        """
        Exchange the current Salesforce access token for a Data Cloud (A360) token.

        Uses the OAuth 2.0 Token Exchange spec:
        POST {instance_url}/services/a360/token
        """
        access_token = self._base.ensure_access()
        assert self._base.instance_url, "instance_url must be set after OAuth access"

        token_exchange_url = f"{self._base.instance_url}/services/a360/token"
        response = requests.post(
            token_exchange_url,
            data={
                "grant_type": "urn:salesforce:grant-type:external:cdp",
                "subject_token": access_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            },
            headers={"Accept": "application/json"},
        )

        logger.info(
            f"A360 token exchange response: status={response.status_code}, elapsed={response.elapsed.total_seconds():.2f}s"
        )

        if response.status_code >= 400:
            logger.error(f"A360 token exchange failed: {response.text}")

        response.raise_for_status()
        payload = response.json()
        # Prefer any instance url present in response, fallback to base
        self.dc_instance_url = (
            payload.get("instance_url")
            or payload.get("a360_instance_url")
            or self._base.instance_url
        )
        return payload

    def ensure_dc_access(self) -> str:
        if self.dc_exp is not None and datetime.now() > self.dc_exp:
            self.dc_exp = None
            self.dc_token = None

        if self.dc_token is None:
            dc_auth_info = self._exchange_for_dc_token()
            self.dc_token = dc_auth_info.get("access_token")
            expires_in = dc_auth_info.get("expires_in")
            if isinstance(expires_in, (int, float)) and expires_in > 0:
                self.dc_exp = datetime.now() + timedelta(seconds=max(60, int(expires_in) - 60))
            else:
                self.dc_exp = datetime.now() + timedelta(minutes=50)

            if not self.dc_token:
                raise RuntimeError("A360 token exchange did not return access_token")

        return self.dc_token

    def get_token(self) -> str:
        return self.ensure_dc_access()

    def get_instance_url(self) -> str:
        # Ensure base is valid and token exchange sets dc_instance_url if needed
        self._base.ensure_access()
        if not self.dc_instance_url:
            self.dc_instance_url = self._base.instance_url
        return self.dc_instance_url  # type: ignore[return-value]

    def get_requests_session(self) -> requests.Session:
        token = self.ensure_dc_access()
        sess = requests.Session()
        sess.headers.update({"Authorization": f"Bearer {token}"})
        return sess


if __name__ == "__main__":
    # Basic CLI to initialize a Data Cloud OAuth session
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(logging.DEBUG)

    cfg = OAuthConfig.from_env()
    base_session = OAuthSession(cfg)
    dc_session = base_session.create_dc_session()

    # Trigger token acquisition (will run interactive OAuth flow if needed)
    dc_session.get_token()
    print(f"Initialized Data Cloud session. Instance URL: {dc_session.get_instance_url()}")
