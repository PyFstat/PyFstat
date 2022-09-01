from subprocess import CalledProcessError

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

    # Test proper return
    for return_output, return_value in zip([True, False], [0, None]):
        assert (
            run_commandline("echo 'Testing returns'", return_output=return_output)
            is return_value
        )

    # Test in case of errors
    with pytest.raises(CalledProcessError):
        run_commandline("ls shrubbery 1>&2", raise_error=True)
    assert caplog.record_tuples[-1][1] == 40
    assert run_commandline("ls shrubbery", raise_error=False) is None
