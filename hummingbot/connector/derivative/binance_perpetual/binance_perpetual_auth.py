import base64
import hashlib
import hmac
import json
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hummingbot.connector.time_synchronizer import TimeSynchronizer
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSRequest


class BinancePerpetualAuth(AuthBase):
    """
    Auth class required by Binance Perpetual API
    """

    _INVALID_ED25519_KEY_ERROR = (
        "Binance API secret must be a valid unencrypted PKCS#8 Ed25519 private key."
    )
    _NON_ASCII_PAYLOAD_ERROR = "Binance authentication payload must contain ASCII characters only."

    def __init__(self, api_key: str, api_secret: str, time_provider: TimeSynchronizer):
        self._api_key: str = api_key
        self._api_secret: str = api_secret
        self._ed25519_private_key: Optional[Ed25519PrivateKey] = self._load_ed25519_private_key(api_secret)
        self._time_provider: TimeSynchronizer = time_provider

    def generate_signature_from_payload(self, payload: str) -> str:
        if not payload.isascii():
            raise ValueError(self._NON_ASCII_PAYLOAD_ERROR)
        encoded_payload = payload.encode("ascii")

        if self._ed25519_private_key is not None:
            signature = self._ed25519_private_key.sign(encoded_payload)
            return base64.b64encode(signature).decode("ascii")

        if self._contains_unicode_surrogate(self._api_secret):
            raise ValueError("Binance API secret must be valid UTF-8 text.")
        secret = self._api_secret.encode("utf-8")
        return hmac.new(secret, encoded_payload, hashlib.sha256).hexdigest()

    @staticmethod
    def _contains_unicode_surrogate(value: str) -> bool:
        return any(0xD800 <= ord(character) <= 0xDFFF for character in value)

    @classmethod
    def _load_ed25519_private_key(cls, api_secret: str) -> Optional[Ed25519PrivateKey]:
        normalized_secret = api_secret.strip()
        if not normalized_secret.startswith("-----BEGIN"):
            return None

        if (
            not normalized_secret.startswith("-----BEGIN PRIVATE KEY-----")
            or not normalized_secret.endswith("-----END PRIVATE KEY-----")
            or normalized_secret.count("-----BEGIN PRIVATE KEY-----") != 1
            or normalized_secret.count("-----END PRIVATE KEY-----") != 1
        ):
            raise ValueError(cls._INVALID_ED25519_KEY_ERROR)
        if not normalized_secret.isascii():
            raise ValueError(cls._INVALID_ED25519_KEY_ERROR)

        private_key = None
        try:
            private_key = serialization.load_pem_private_key(
                normalized_secret.encode("ascii"),
                password=None,
            )
        except (TypeError, ValueError, UnicodeEncodeError, UnsupportedAlgorithm):
            pass

        if private_key is None or not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(cls._INVALID_ED25519_KEY_ERROR)

        return private_key

    async def rest_authenticate(self, request: RESTRequest) -> RESTRequest:
        if request.method == RESTMethod.POST:
            request.data = self.add_auth_to_params(params=json.loads(request.data) if request.data is not None else {})
        else:
            request.params = self.add_auth_to_params(request.params)

        request.headers = self.header_for_authentication()

        return request

    async def ws_authenticate(self, request: WSRequest) -> WSRequest:
        return request  # pass-through

    def add_auth_to_params(self,
                           params: Dict[str, Any]):
        timestamp = int(self._time_provider.time() * 1e3)

        request_params = OrderedDict(params or {})
        request_params["timestamp"] = timestamp

        payload = urlencode(request_params)
        request_params["signature"] = self.generate_signature_from_payload(payload=payload)

        return request_params

    def header_for_authentication(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self._api_key}
