from ... import utils
from ...logger import log
from astropy import time
from pathlib import Path
import requests

V3GR_ARCHIVE_URL: str = (
    "https://vmf.geo.tuwien.ac.at/trop_products/VLBI/V3GR/V3GR_OP/daily/"
)
"""URL of archive from which to download V3GR files"""


def get_v3gr_url_for_epoch(epoch: "time.Time") -> str:
    """URL to download V3GR file for a given epoch

    Extracts the year, and day of the year from the date, and returns the URL from which to download the corresponding V3GR file.

    :param epoch: Epoch in UTC
    :return url: URL to download the V3GR file for the given date
    """

    # Get year and day of year for the given date
    year = utils.get_year_from_epoch(epoch)
    day_of_year = utils.get_day_of_year_from_epoch(epoch)

    # Define file name and url
    file_name = f"{year:04d}{day_of_year:03d}.v3gr_r"
    url = V3GR_ARCHIVE_URL + f"{year:04d}/{file_name}"

    return url


def download_v3gr_file(url: str, output_directory: str | Path) -> Path:
    """Download V3GR file from URL

    :param url: URL to download the V3GR file
    :param output_directory: Directory to save the downloaded file
    """

    # Turn inputs into Path objects to extract components
    v3gr_path = Path(url)
    output_directory = Path(output_directory)
    output_file = output_directory / v3gr_path.name

    # Create the output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    # Download the file if it doesn't exist
    if not output_file.exists():

        log.info(f"Downloading V3GR file: {output_file.name}")

        # Download content of URL
        response = requests.get(url)
        if not response.ok:
            log.error(f"Failed to download V3GR file: {url}")
            exit(1)

        # Write the content to the output file
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(response.content)
    else:
        log.debug(f"Found {output_file}")

    return output_file


def download_v3gr_file_for_epoch(
    date: "time.Time", output_directory: str | Path
) -> Path:
    """Download V3GR file for a given date

    The function finds the URL of the V3GR file for the given epoch, and downloads it to the specified directory. The directory is created if it doesn't exist.

    :param epoch: Epoch in UTC
    :param output_directory: Directory to save the downloaded file
    """

    # Get URL for the given date
    v3gr_url = get_v3gr_url_for_epoch(date)

    # Download the V3GR file
    v3gr_file = download_v3gr_file(v3gr_url, output_directory)

    return v3gr_file
