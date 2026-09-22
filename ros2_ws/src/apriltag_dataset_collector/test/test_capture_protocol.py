from apriltag_dataset_collector.capture_protocol import (
    build_capture_request,
    normalize_dataset_root,
    parse_nullable_float,
)


def test_nullable_truth_values_are_preserved():
    request = build_capture_request(
        session_id="pool_a",
        operator_name=" recorder ",
        tag_id="",
        truth={"x_m": "1.25", "y_m": "", "z_m": "-0.5", "yaw_deg": "90"},
        note=" first point ",
        request_id="request_a",
    )

    assert request["request_id"] == "request_a"
    assert request["operator"] == "recorder"
    assert request["note"] == "first point"
    assert "tag_id" not in request
    assert request["truth_world"]["x_m"] == 1.25
    assert request["truth_world"]["y_m"] is None
    assert request["truth_world"]["z_m"] == -0.5
    assert request["truth_world"]["yaw_deg"] == 90.0


def test_tag_and_empty_session_are_normalized():
    request = build_capture_request(session_id="", tag_id="15", request_id="request_b")

    assert request["request_id"] == "request_b"
    assert request["tag_id"] == 15
    assert request["session_id"]


def test_dataset_root_and_null_parser_validation():
    assert parse_nullable_float(" null ") is None
    assert normalize_dataset_root(" ~/datasets ").endswith("datasets")
