from decimal import Decimal
from unittest import TestCase
from unittest.mock import patch

from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative import BinancePerpetualDerivative
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState


class ExchangePyBasePreallocatedClientOrderIdTest(TestCase):

    trading_pair = "SNDK-USDT"

    def setUp(self) -> None:
        super().setUp()
        self.exchange = BinancePerpetualDerivative(
            binance_perpetual_api_key="test-api-key",
            binance_perpetual_api_secret="test-secret",
            trading_pairs=[self.trading_pair],
            trading_required=False,
        )
        self.scheduled_order_ids = []
        self.schedule_patch = patch(
            "hummingbot.connector.exchange_py_base.safe_ensure_future",
            side_effect=self._capture_and_close_order,
        )
        self.mock_schedule = self.schedule_patch.start()

    def tearDown(self) -> None:
        self.schedule_patch.stop()
        super().tearDown()

    def _capture_and_close_order(self, order_coroutine):
        self.scheduled_order_ids.append(order_coroutine.cr_frame.f_locals["order_id"])
        order_coroutine.close()

    def _tracked_order(self, client_order_id: str, state: OrderState) -> InFlightOrder:
        return InFlightOrder(
            client_order_id=client_order_id,
            exchange_order_id="10001",
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1"),
            price=Decimal("10"),
            creation_timestamp=1,
            initial_state=state,
        )

    def test_buy_and_sell_preserve_caller_preallocated_ids_without_generation(self):
        buy_id = "exec-sndk-snxx-0001-maker-0"
        sell_id = "exec-sndk-snxx-0001-stock-0"

        with patch("hummingbot.connector.exchange_py_base.get_new_client_order_id") as generate_id:
            returned_buy_id = self.exchange.buy(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
                order_type=OrderType.LIMIT_MAKER,
                price=Decimal("10"),
                client_order_id=buy_id,
            )
            returned_sell_id = self.exchange.sell(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
                order_type=OrderType.MARKET,
                client_order_id=sell_id,
            )

        self.assertEqual(buy_id, returned_buy_id)
        self.assertEqual(sell_id, returned_sell_id)
        self.assertEqual([buy_id, sell_id], self.scheduled_order_ids)
        generate_id.assert_not_called()

    def test_default_callers_keep_generated_client_order_id_behavior(self):
        generated_ids = ["x-nbQe1H39-B-SNDK-1", "x-nbQe1H39-S-SNDK-2"]

        with patch(
            "hummingbot.connector.exchange_py_base.get_new_client_order_id",
            side_effect=generated_ids,
        ) as generate_id:
            returned_buy_id = self.exchange.buy(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
            )
            returned_sell_id = self.exchange.sell(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
            )

        self.assertEqual(generated_ids, [returned_buy_id, returned_sell_id])
        self.assertEqual(generated_ids, self.scheduled_order_ids)
        self.assertEqual(2, generate_id.call_count)

    def test_invalid_preallocated_ids_fail_synchronously_without_scheduling(self):
        invalid_ids = (
            False,
            1,
            b"bytes-id",
            "",
            " ",
            "contains?question",
            "contains unicode 雪",
            "a" * (self.exchange.client_order_id_max_length + 1),
            f"{self.exchange.client_order_id_prefix}-caller-owned",
        )

        for invalid_id in invalid_ids:
            with self.subTest(client_order_id=invalid_id):
                with self.assertRaises((TypeError, ValueError)):
                    self.exchange.buy(
                        trading_pair=self.trading_pair,
                        amount=Decimal("1"),
                        client_order_id=invalid_id,
                    )

        self.mock_schedule.assert_not_called()

    def test_exact_connector_length_and_exchange_characters_are_accepted(self):
        client_order_id = "." + "a" * (self.exchange.client_order_id_max_length - 5) + ":/_-"

        returned_id = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            client_order_id=client_order_id,
        )

        self.assertEqual(self.exchange.client_order_id_max_length, len(client_order_id))
        self.assertEqual(client_order_id, returned_id)
        self.assertEqual([client_order_id], self.scheduled_order_ids)

    def test_duplicate_preallocated_id_is_rejected_before_tracker_can_start_it(self):
        client_order_id = "exec-sndk-snxx-0002-stock-0"

        self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            client_order_id=client_order_id,
        )

        with self.assertRaisesRegex(ValueError, "already reserved"):
            self.exchange.sell(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
                client_order_id=client_order_id,
            )

        self.assertEqual([client_order_id], self.scheduled_order_ids)

    def test_active_cached_and_lost_tracker_collisions_are_rejected(self):
        active_id = "exec-sndk-snxx-0003-maker-0"
        cached_id = "exec-sndk-snxx-0003-stock-0"
        lost_id = "exec-sndk-snxx-0003-rollback-0"
        self.exchange._order_tracker.start_tracking_order(
            self._tracked_order(active_id, OrderState.OPEN)
        )
        self.exchange._order_tracker.start_tracking_order(
            self._tracked_order(cached_id, OrderState.CANCELED)
        )
        self.exchange._order_tracker.stop_tracking_order(cached_id)
        self.exchange._order_tracker._lost_orders[lost_id] = self._tracked_order(
            lost_id, OrderState.FAILED
        )

        for colliding_id in (active_id, cached_id, lost_id):
            with self.subTest(client_order_id=colliding_id):
                with self.assertRaisesRegex(ValueError, "order tracker"):
                    self.exchange.buy(
                        trading_pair=self.trading_pair,
                        amount=Decimal("1"),
                        client_order_id=colliding_id,
                    )

        self.mock_schedule.assert_not_called()

    def test_terminal_preallocated_id_remains_single_use_for_connector_lifetime(self):
        client_order_id = "exec-sndk-snxx-0004-maker-0"

        self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            client_order_id=client_order_id,
        )
        self.exchange._order_tracker.start_tracking_order(
            self._tracked_order(client_order_id, OrderState.FILLED)
        )
        self.exchange._order_tracker.stop_tracking_order(client_order_id)
        del self.exchange._order_tracker._cached_orders[client_order_id]

        with self.assertRaisesRegex(ValueError, "already reserved"):
            self.exchange.buy(
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
                client_order_id=client_order_id,
            )

        self.assertEqual([client_order_id], self.scheduled_order_ids)
