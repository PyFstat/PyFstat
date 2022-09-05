from subprocess import CalledProcessError, CompletedProcess

import pytest

from pyfstat.utils import run_commandline


def test_run_commandline_noreturn():
    assert run_commandline("echo 'Testing no return'", return_output=False) is None


def test_run_commandline_goodreturn(caplog):
    return_obj = run_commandline("echo 'Testing return'", return_output=True)
    _, log_level, log_message = caplog.record_tuples[-1]
    assert type(return_obj) is CompletedProcess
    assert log_level == 20
    assert return_obj.stdout == log_message + "\n"


def test_run_commandline_stdout_stderr(caplog):
    # Test warning for pipes and proper stdout and stderr redirect
    msg = "This is a simulated error message, do not worry human!"
    run_commandline(
        f"echo '{msg}' | tee /dev/stderr"
    )  # print this to both stdout and stderr
    # check that the messages came through at the right logging levels
    # 20 == INFO, 30 == WARNING, 40 == ERROR
    messages = caplog.record_tuples[-3:]
    assert (messages[0][1] == 30) and (
        "Pipe ('|') found in commandline" in messages[0][2]
    )
    assert messages[1][1] == 20
    assert messages[1][2] == msg
    assert messages[2][1] == 40
    assert messages[2][2] == msg


def test_run_commandline_expected_error_with_raise(caplog):
    with pytest.raises(CalledProcessError):
        run_commandline("ls nonexistent-directory 1>&2", raise_error=True)
    assert caplog.record_tuples[-1][1] == 40


def test_run_commandline_expected_error_noraise(caplog):
    assert run_commandline("ls nonexistent-directory", raise_error=False) is None


def test_run_commandline_lal_noerror(caplog):
    # Basic print should come out as INFO
    return_obj = run_commandline("lalpulsar_version", return_output=True)
    assert return_obj.stderr == ""
    assert return_obj.returncode == 0
    for return_line, (_, log_level, log_message) in zip(
        return_obj.stdout.split("\n"), caplog.record_tuples[1:]
    ):
        assert log_level == 20
        assert return_line == log_message


def test_run_commandline_lal_expected_error(caplog):
    # Errors should go through stderr and come out as ERROR
    with pytest.raises(CalledProcessError):
        return_obj = run_commandline(
            "lalpulsar_version --nonexistent-option-for-simulated-error",
            return_output=True,
        )
        _, log_level, log_message = caplog.record_tuples[-1]
        assert log_level == 40
        assert return_obj.stderr == log_message
        assert return_obj.stdout == ""
        assert return_obj.returncode != 0


def test_run_commandline_lal_expected_error_and_stdout(caplog):
    # this one should print something to stdout *before* the error
    run_commandline(
        "lalpulsar_ComputeFstatistic_v2 --DataFiles no_such_file --Alpha 0 --Delta 0",
        raise_error=False,
    )
    for _, log_level, log_message in caplog.record_tuples:
        if "Now executing:" in log_message or "[normal]" in log_message:
            assert log_level == 20
        else:
            assert log_level == 40
