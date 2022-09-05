from subprocess import CalledProcessError, CompletedProcess

import pytest

import pyfstat
from pyfstat.helper_functions import run_commandline


def test_gps_to_datestr_utc():

    gps = 1000000000
    # reference from lal_tconvert on en_US.UTF-8 locale
    # but instead from the "GMT" bit it places in the middle,
    # we put the timezone info at the end via datetime.strftime()
    old_str = "Wed Sep 14 01:46:25 GMT 2011"
    new_str = pyfstat.helper_functions.gps_to_datestr_utc(gps)
    assert new_str.rstrip(" UTC") == old_str.replace(" GMT ", " ")


def test_run_commandline(caplog):

    # Test warning for pipes and proper stdout and stderr redirect
    msg = "Just a flesh wound"
    run_commandline(f"echo '{msg}' | tee /dev/stderr")

    messages = caplog.record_tuples[-3:]
    assert (messages[0][1] == 30) and (
        "Pipe ('|') found in commandline" in messages[0][2]
    )
    assert (messages[1][1] == 20) and (messages[1][2] == msg + "\n")
    assert (messages[2][1] == 40) and (messages[2][2] == msg + "\n")

    # Test proper return without errors
    return_obj = run_commandline(
        "echo 'Testing return' | tee /dev/stderr", return_output=True
    )
    messages = caplog.record_tuples[-2:]
    assert type(return_obj) is CompletedProcess
    assert (messages[0][1] == 20) and (return_obj.stdout == messages[0][2])
    assert (messages[1][1] == 40) and (return_obj.stderr == messages[1][2])

    assert run_commandline("echo 'Testing no return'", return_output=False) is None

    # Test in case of errors
    with pytest.raises(CalledProcessError):
        run_commandline("ls shrubbery 1>&2", raise_error=True)
    assert caplog.record_tuples[-1][1] == 40
    assert run_commandline("ls shrubbery", raise_error=False) is None


def test_run_commandline_on_lal(caplog):

    # Basic print should come out as INFO
    return_obj = run_commandline("lalpulsar_version", return_output=True)

    _, log_level, log_message = caplog.record_tuples[-1]
    assert log_level == 20
    assert return_obj.stdout == log_message
    assert return_obj.stderr == ""
    assert return_obj.returncode == 0

    # Errors should go through stderr and come out as ERROR
    with pytest.raises(CalledProcessError):
        return_obj = run_commandline("lalpulsar_Makefakedata_v5", return_output=True)
        _, log_level, log_message = caplog.record_tuples[-1]
        assert log_level == 40
        assert return_obj.stderr == log_message
        assert return_obj.stdout == ""
        assert return_obj.returncode != 0
