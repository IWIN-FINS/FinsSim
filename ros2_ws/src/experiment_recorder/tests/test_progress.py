from experiment_recorder.progress import log_stage


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_log_stage_uses_standard_ros_info_message_without_nested_prefix() -> None:
    logger = _Logger()
    log_stage("T1 acquisition 2/8: recorder active", logger=logger)
    assert logger.messages == ["T1 acquisition 2/8: recorder active"]


def test_log_stage_marks_standalone_analysis_output_as_info(capsys) -> None:
    log_stage("analysis processing: session=trial-01")
    assert capsys.readouterr().out == "[INFO] analysis processing: session=trial-01\n"
