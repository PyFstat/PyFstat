import logging
import os

logger = logging.getLogger(__name__)


def get_ephemeris_files():
    """Set the ephemeris files to use for the Earth and Sun.

    This looks first for a configuration file `~/.pyfstat.conf`
    giving individual earth/sun file paths like this:

    ```
    earth_ephem = '/my/path/earth00-40-DE405.dat.gz'
    sun_ephem = '/my/path/sun00-40-DE405.dat.gz'
    ```

    If such a file is not found or does not conform to that format,
    then we rely on lal's recently improved ability to find proper
    default fallback paths for the `[earth/sun]00-40-DE405` ephemerides
    with both pip- and conda-installed packages,

    Alternatively, ephemeris options can be set manually
    on each class instantiation.

    NOTE that the `$LALPULSAR_DATADIR` environment variable
    is no longer supported!

    Returns
    ----------
    earth_ephem, sun_ephem: str
        Paths of the two files containing positions of Earth and Sun.
    """
    config_file = os.path.join(os.path.expanduser("~"), ".pyfstat.conf")
    ephem_version = "DE405"
    earth_ephem = f"earth00-40-{ephem_version}.dat.gz"
    sun_ephem = f"sun00-40-{ephem_version}.dat.gz"
    please = "Will fall back to lal's automatic path resolution for files"
    please += f" [{earth_ephem},{sun_ephem}]."
    please += " Alternatively, set 'earth_ephem' and 'sun_ephem' class options."
    if os.path.isfile(config_file):
        d = {}
        with open(config_file, "r") as f:
            for line in f:
                k, v = line.split("=")
                k = k.replace(" ", "")
                for item in [" ", "'", '"', "\n"]:
                    v = v.replace(item, "")
                d[k] = v
        try:
            earth_ephem = d["earth_ephem"]
            sun_ephem = d["sun_ephem"]
        except KeyError:
            logger.warning(f"No [earth/sun]_ephem found in {config_file}. {please}")
    else:
        logger.info(f"No {config_file} file found. {please}")
    return earth_ephem, sun_ephem
