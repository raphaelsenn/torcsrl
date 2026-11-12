import pytest

from gym_torcs.client import Action, parse_sensors


class TestParseSensors:
    def test_single_values_become_floats(self) -> None:
        sensors = parse_sensors("(angle 0.01)(speedX 120.5)")
        assert sensors == {"angle": 0.01, "speedX": 120.5}

    def test_multiple_values_become_lists(self) -> None:
        sensors = parse_sensors("(track 1 2 3)(wheelSpinVel 4 5 6 7)")
        assert sensors["track"] == [1.0, 2.0, 3.0]
        assert sensors["wheelSpinVel"] == [4.0, 5.0, 6.0, 7.0]

    def test_ignores_null_padding_and_whitespace(self) -> None:
        sensors = parse_sensors(" (gear 3)(rpm 4000)\x00\x00")
        assert sensors == {"gear": 3.0, "rpm": 4000.0}

    def test_skips_items_without_a_value(self) -> None:
        assert parse_sensors("(empty)(rpm 1)") == {"rpm": 1.0}


class TestAction:
    def test_default_action_idles_in_first_gear(self) -> None:
        message = Action().encode().decode()
        assert "(accel 0.000)" in message
        assert "(brake 0.000)" in message
        assert "(gear 1)" in message
        assert "(steer 0.000)" in message

    def test_encodes_values_with_three_decimals(self) -> None:
        message = Action(steer=-0.12345, accel=1.0, gear=4).encode().decode()
        assert "(steer -0.123)" in message
        assert "(accel 1.000)" in message
        assert "(gear 4)" in message

    def test_encoded_action_is_bytes(self) -> None:
        assert isinstance(Action().encode(), bytes)


@pytest.mark.parametrize("message", ["(angle 1)", "(a 1)(b 2 3)"])
def test_parse_round_trip_keeps_all_names(message: str) -> None:
    names = {item.split()[0] for item in message.strip("()").split(")(")}
    assert set(parse_sensors(message)) == names
