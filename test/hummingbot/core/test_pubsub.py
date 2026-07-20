import asyncio
import gc
import unittest
import weakref
from test.mock.mock_events import MockEvent, MockEventType

from hummingbot.core.event.event_forwarder import EventForwarder
from hummingbot.core.event.event_logger import EventLogger
from hummingbot.core.pubsub import PubSub


class DeliberateBaseException(BaseException):
    pass


class PubSubTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pubsub = PubSub()
        self.listener_zero = EventLogger()
        self.listener_one = EventLogger()
        self.event_tag_zero = MockEventType.EVENT_ZERO
        self.event_tag_one = MockEventType.EVENT_ONE
        self.event = MockEvent(payload=1)

    def test_get_listeners_no_listeners(self):
        listeners_count = len(self.pubsub.get_listeners(self.event_tag_zero))
        self.assertEqual(0, listeners_count)

    def test_add_listeners(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        listeners = self.pubsub.get_listeners(self.event_tag_zero)
        self.assertEqual(1, len(listeners))
        self.assertIn(self.listener_zero, listeners)

        self.pubsub.add_listener(self.event_tag_zero, self.listener_one)
        listeners = self.pubsub.get_listeners(self.event_tag_zero)
        self.assertEqual(2, len(listeners))
        self.assertIn(self.listener_zero, listeners)
        self.assertIn(self.listener_one, listeners)

    def test_add_listener_twice(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        listeners_count = len(self.pubsub.get_listeners(self.event_tag_zero))
        self.assertEqual(1, listeners_count)

        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        listeners_count = len(self.pubsub.get_listeners(self.event_tag_zero))
        self.assertEqual(1, listeners_count)

    def test_remove_listener(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        self.pubsub.add_listener(self.event_tag_zero, self.listener_one)

        self.pubsub.remove_listener(self.event_tag_zero, self.listener_zero)
        listeners = self.pubsub.get_listeners(self.event_tag_zero)
        self.assertNotIn(self.listener_zero, listeners)
        self.assertIn(self.listener_one, listeners)

    def test_add_listeners_to_separate_events(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        self.pubsub.add_listener(self.event_tag_one, self.listener_one)

        listeners_zero = self.pubsub.get_listeners(self.event_tag_zero)
        listeners_one = self.pubsub.get_listeners(self.event_tag_one)
        self.assertEqual(1, len(listeners_zero))
        self.assertEqual(1, len(listeners_one))

    def test_trigger_event(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        self.pubsub.add_listener(self.event_tag_one, self.listener_one)
        self.pubsub.trigger_event(self.event_tag_zero, self.event)
        self.assertEqual(1, len(self.listener_zero.event_log))
        self.assertEqual(self.event, self.listener_zero.event_log[0])
        self.assertEqual(0, len(self.listener_one.event_log))

    def test_trigger_event_continues_after_synchronous_listener_cancelled_error(self):
        observations = []

        for cancel_first in (True, False):
            pubsub = PubSub()
            recording_listener = EventLogger()
            cancellation_calls = []

            def cancel_synchronously(event):
                cancellation_calls.append(event)
                raise asyncio.CancelledError("listener-local cancellation")

            canceling_listener = EventForwarder(cancel_synchronously)
            listeners = (
                (canceling_listener, recording_listener)
                if cancel_first
                else (recording_listener, canceling_listener)
            )
            for listener in listeners:
                pubsub.add_listener(self.event_tag_zero, listener)

            error_name = None
            try:
                pubsub.trigger_event(self.event_tag_zero, self.event)
            except BaseException as error:
                error_name = type(error).__name__
            observations.append((
                cancel_first,
                error_name,
                cancellation_calls,
                recording_listener.event_log,
            ))

        self.assertEqual(
            [
                (cancel_first, None, [self.event], [self.event])
                for cancel_first in (True, False)
            ],
            observations,
        )

    def test_trigger_event_does_not_swallow_other_base_exceptions(self):
        for exception_type in (SystemExit, KeyboardInterrupt, DeliberateBaseException):
            with self.subTest(exception_type=exception_type):
                pubsub = PubSub()

                def raise_base_exception(event):
                    raise exception_type("must propagate")

                listener = EventForwarder(raise_base_exception)
                pubsub.add_listener(self.event_tag_zero, listener)

                with self.assertRaises(exception_type):
                    pubsub.trigger_event(self.event_tag_zero, self.event)

    def test_lapsed_listener_remove_on_get_listeners(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        self.listener_zero = None  # remove strong reference
        gc.collect()
        listeners = self.pubsub.get_listeners(self.event_tag_zero)
        self.assertEqual(0, len(listeners))

    def test_lapsed_listener_remove_on_remove_listener(self):
        self.pubsub.add_listener(self.event_tag_zero, self.listener_zero)
        self.pubsub.add_listener(self.event_tag_zero, self.listener_one)
        listener_zero_weakref = weakref.ref(self.listener_zero)
        listener_one_weakref = weakref.ref(self.listener_one)
        listeners = None
        self.listener_zero = None  # remove strong reference
        gc.collect()
        self.pubsub.remove_listener(self.event_tag_zero, self.listener_one)
        self.assertEqual(None, listener_zero_weakref())
        self.assertNotEqual(None, listener_one_weakref())
        listeners = self.pubsub.get_listeners(self.event_tag_zero)
        self.assertEqual(0, len(listeners))


if __name__ == "__main__":
    unittest.main()
