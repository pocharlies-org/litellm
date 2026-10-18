import base64
import json
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"


class TestDeviceLoginMode:
    """CHATGPT_DEVICE_LOGIN: a server never blocks waiting for a human (DGX-1053)."""

    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def _no_token(self):
        return patch("builtins.open", side_effect=OSError("missing"))

    def test_never_raises_instead_of_device_login(self, authenticator, monkeypatch):
        from litellm.llms.chatgpt.common_utils import GetAccessTokenError

        monkeypatch.setenv("CHATGPT_DEVICE_LOGIN", "never")
        with self._no_token(), patch.object(authenticator, "_login_device_code") as login:
            with pytest.raises(GetAccessTokenError, match="device login disabled"):
                authenticator.get_access_token()
        login.assert_not_called()

    def test_auto_without_tty_behaves_as_never(self, authenticator, monkeypatch):
        from litellm.llms.chatgpt.common_utils import GetAccessTokenError

        monkeypatch.delenv("CHATGPT_DEVICE_LOGIN", raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
        with self._no_token(), patch.object(authenticator, "_login_device_code") as login:
            with pytest.raises(GetAccessTokenError):
                authenticator.get_access_token()
        login.assert_not_called()

    def test_auto_with_tty_and_always_keep_device_login(self, authenticator, monkeypatch):
        for mode, tty in (("auto", True), ("always", False)):
            monkeypatch.setenv("CHATGPT_DEVICE_LOGIN", mode)
            monkeypatch.setattr("sys.stdin.isatty", lambda tty=tty: tty, raising=False)
            with self._no_token(), patch.object(
                authenticator, "_login_device_code", return_value={"access_token": "t"}
            ) as login, patch.object(authenticator, "_get_device_code_cooldown_remaining", return_value=0):
                assert authenticator.get_access_token() == "t"
            login.assert_called_once()
