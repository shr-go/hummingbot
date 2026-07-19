import asyncio
import copy
import hashlib
import hmac
import json
import unittest
from typing import Awaitable
from urllib.parse import urlencode

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_auth import BinancePerpetualAuth
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSJSONRequest


ED25519_PKCS8_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEILcA8v5gDH7XncFCLnshzaIqicdNSANBOQvrpNQNRiie
-----END PRIVATE KEY-----
"""
ED25519_FIXED_SIGNATURE = "EJxg47AJGKaCTkIUQzJc1WYo7TyJwgxTTVIOMI2QzHE8h1670jajm7/dLsjpr4i/Inlpi/OFNGZIhsmCp1U9CQ=="


class BinancePerpetualAuthUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.ev_loop = asyncio.get_event_loop()
        cls.api_key = "TEST_API_KEY"
        cls.secret_key = "TEST_SECRET_KEY"

    def setUp(self) -> None:
        super().setUp()
        self.emulated_time = 1640001112.223
        self.test_params = {
            "test_param": "test_input",
            "timestamp": int(self.emulated_time * 1e3),
        }
        self.auth = BinancePerpetualAuth(
            api_key=self.api_key,
            api_secret=self.secret_key,
            time_provider=self)

    def _get_test_payload(self):
        return urlencode(dict(copy.deepcopy(self.test_params)))

    def _get_signature_from_test_payload(self):
        return hmac.new(
            bytes(self.auth._api_secret.encode("utf-8")), self._get_test_payload().encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def async_run_with_timeout(self, coroutine: Awaitable, timeout: float = 1):
        ret = self.ev_loop.run_until_complete(asyncio.wait_for(coroutine, timeout))
        return ret

    def time(self):
        # Implemented to emulate a TimeSynchronizer
        return self.emulated_time

    def test_generate_signature_from_payload_uses_hmac_for_plain_secret(self):
        payload = self._get_test_payload()
        signature = self.auth.generate_signature_from_payload(payload)

        self.assertEqual(signature, self._get_signature_from_test_payload())

    def test_generate_signature_from_payload_with_ed25519_pkcs8_matches_fixed_vector(self):
        auth = BinancePerpetualAuth(
            api_key=self.api_key,
            api_secret=ED25519_PKCS8_PRIVATE_KEY,
            time_provider=self,
        )

        signature = auth.generate_signature_from_payload(self._get_test_payload())

        self.assertEqual(ED25519_FIXED_SIGNATURE, signature)

    def test_generate_signature_from_payload_accepts_multiline_ed25519_pem(self):
        multiline_secret = f"\n{ED25519_PKCS8_PRIVATE_KEY}\n"
        auth = BinancePerpetualAuth(
            api_key=self.api_key,
            api_secret=multiline_secret,
            time_provider=self,
        )

        signature = auth.generate_signature_from_payload(self._get_test_payload())

        self.assertEqual(ED25519_FIXED_SIGNATURE, signature)

    def test_generate_signature_from_payload_accepts_crlf_ed25519_pem(self):
        crlf_secret = ED25519_PKCS8_PRIVATE_KEY.replace("\n", "\r\n")
        auth = BinancePerpetualAuth(
            api_key=self.api_key,
            api_secret=crlf_secret,
            time_provider=self,
        )

        signature = auth.generate_signature_from_payload(self._get_test_payload())

        self.assertEqual(ED25519_FIXED_SIGNATURE, signature)

    def test_ed25519_secret_rejects_malformed_pkcs8_pem(self):
        malformed_secrets = (
            "-----BEGIN PRIVATE KEY-----\nnot-valid-base64!\n-----END PRIVATE KEY-----",
            "-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIA==",
        )

        for malformed_secret in malformed_secrets:
            with self.subTest(malformed_secret=malformed_secret):
                with self.assertRaisesRegex(ValueError, "PKCS#8 Ed25519"):
                    BinancePerpetualAuth(
                        api_key=self.api_key,
                        api_secret=malformed_secret,
                        time_provider=self,
                    )

    def test_ed25519_secret_rejects_non_ed25519_pkcs8_key(self):
        ec_private_key = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")

        with self.assertRaisesRegex(ValueError, "PKCS#8 Ed25519"):
            BinancePerpetualAuth(
                api_key=self.api_key,
                api_secret=ec_private_key,
                time_provider=self,
            )

    def test_ed25519_secret_rejects_encrypted_pkcs8_key(self):
        private_key = serialization.load_pem_private_key(
            ED25519_PKCS8_PRIVATE_KEY.encode("ascii"),
            password=None,
        )
        encrypted_private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(b"test-password"),
        ).decode("ascii")

        with self.assertRaisesRegex(ValueError, "PKCS#8 Ed25519"):
            BinancePerpetualAuth(
                api_key=self.api_key,
                api_secret=encrypted_private_key,
                time_provider=self,
            )

    def test_authentication_errors_redact_api_key_and_private_key(self):
        api_key = "API_KEY_MUST_NOT_LEAK"
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            "PRIVATE_KEY_BODY_MUST_NOT_LEAK\n"
            "-----END PRIVATE KEY-----"
        )

        with self.assertRaises(ValueError) as exception_context:
            BinancePerpetualAuth(
                api_key=api_key,
                api_secret=private_key,
                time_provider=self,
            )

        error = repr(exception_context.exception)
        self.assertNotIn(api_key, error)
        self.assertNotIn(private_key, error)
        self.assertNotIn("PRIVATE_KEY_BODY_MUST_NOT_LEAK", error)

    def test_generate_signature_from_payload_rejects_unicode(self):
        for secret in (self.secret_key, ED25519_PKCS8_PRIVATE_KEY):
            with self.subTest(secret_type="pem" if secret.startswith("-----BEGIN") else "hmac"):
                auth = BinancePerpetualAuth(
                    api_key=self.api_key,
                    api_secret=secret,
                    time_provider=self,
                )

                with self.assertRaisesRegex(ValueError, "ASCII"):
                    auth.generate_signature_from_payload("symbol=\u6d4b\u8bd5")

    def test_rest_authenticate_parameters_provided(self):
        request: RESTRequest = RESTRequest(
            method=RESTMethod.GET, url="/TEST_PATH_URL", params=copy.deepcopy(self.test_params), is_auth_required=True
        )

        signed_request: RESTRequest = self.async_run_with_timeout(self.auth.rest_authenticate(request))

        self.assertIn("X-MBX-APIKEY", signed_request.headers)
        self.assertEqual(signed_request.headers["X-MBX-APIKEY"], self.api_key)
        self.assertIn("signature", signed_request.params)
        self.assertEqual(signed_request.params["signature"], self._get_signature_from_test_payload())

    def test_rest_authenticate_data_provided(self):
        request: RESTRequest = RESTRequest(
            method=RESTMethod.POST, url="/TEST_PATH_URL", data=json.dumps(self.test_params), is_auth_required=True
        )

        signed_request: RESTRequest = self.async_run_with_timeout(self.auth.rest_authenticate(request))

        self.assertIn("X-MBX-APIKEY", signed_request.headers)
        self.assertEqual(signed_request.headers["X-MBX-APIKEY"], self.api_key)
        self.assertIn("signature", signed_request.data)
        self.assertEqual(signed_request.data["signature"], self._get_signature_from_test_payload())

    def test_rest_authenticate_uses_ed25519_signature(self):
        auth = BinancePerpetualAuth(
            api_key=self.api_key,
            api_secret=ED25519_PKCS8_PRIVATE_KEY,
            time_provider=self,
        )
        request: RESTRequest = RESTRequest(
            method=RESTMethod.GET,
            url="/TEST_PATH_URL",
            params=copy.deepcopy(self.test_params),
            is_auth_required=True,
        )

        signed_request: RESTRequest = self.async_run_with_timeout(auth.rest_authenticate(request))

        self.assertEqual(ED25519_FIXED_SIGNATURE, signed_request.params["signature"])

    def test_ws_authenticate(self):
        request: WSJSONRequest = WSJSONRequest(
            payload={"TEST": "SOME_TEST_PAYLOAD"}, throttler_limit_id="TEST_LIMIT_ID", is_auth_required=True
        )

        signed_request: WSJSONRequest = self.async_run_with_timeout(self.auth.ws_authenticate(request))

        self.assertEqual(request, signed_request)
