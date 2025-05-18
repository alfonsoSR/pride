from pathlib import Path
from ...logger import log
from ftplib import FTP_TLS
import unlzw3
import gzip
from astropy import time
from ... import utils

IGS_NEW_FORMAT_GPS_WEEK: int = 2238
"""GPS week for the new IONEX file format"""
IONEX_FTP_SERVER: str = "gdc.cddis.eosdis.nasa.gov"
"""FTP server for IONEX files"""


def get_ionex_path_for_date(date: "time.Time") -> str:
    """Path to IONEX file in FTP for a given date

    Extracts the year, GPS week, and day of year from the given date, and returns the path to the corresponding IONEX file in the FTP server. The function selects the correct convention for the file name based on the date.

    :param date: Date in UTC. Must be at 00:00:00 UTC.
    :return file_name: Path to the IONEX file for the given date
    """

    # Ensure that the input is a date, not date and time
    if (date - utils.get_date_from_epoch(date)).to_value("s") != 0:
        log.error(
            f"Failed to get URL for IONEX file for {date.iso}: "
            "Date should be at 00:00:00 UTC"
        )
        exit(1)

    # Get GPS week, year, and day of year for the given date
    gps_week = utils.get_gps_week_for_date(date)
    year = utils.get_year_from_epoch(date)
    day_of_year = utils.get_day_of_year_from_epoch(date)

    # Define file name based on the GPS week
    if gps_week < IGS_NEW_FORMAT_GPS_WEEK:
        file_name = f"igsg{day_of_year:03d}0.{str(year)[2:]}i.Z"
    else:
        file_name = (
            f"IGS0OPSFIN_{year:04d}{day_of_year:03d}0000_01D_02H_GIM.INX.gz"
        )

    # Define the path to the file on the FTP server
    file_path = f"{year:4d}/{day_of_year:03d}/{file_name}"

    return file_path


def download_compressed_ionex_file(
    ftp_path: str, output_directory: str | Path
) -> Path:
    """Download compressed IONEX file from FTP server

    :param ftp_path: Path to the IONEX file on the FTP server
    :param output_directory: Directory to save the downloaded file
    """

    # Turn inputs into Path objects to extract components
    ionex_path = Path(ftp_path)
    output_directory = Path(output_directory)
    output_file = output_directory / ionex_path.name

    # If the output file already exists, skip the download
    if output_file.exists():
        log.debug(f"Found {output_file}")
        return output_file

    # Also check if the decompressed file exists
    if output_file.with_suffix("").exists():
        log.debug(f"Found {output_file.with_suffix('')}")
        return output_file.with_suffix("")

    log.info(f"Downloading IONEX file: {ionex_path.name}")

    # Initialize FTP connection
    ftp_connection = FTP_TLS(IONEX_FTP_SERVER)
    ftp_connection.login(user="anonymous", passwd="")
    ftp_connection.prot_p()

    # Go to parent directory of the file
    ftp_connection.cwd(f"gps/products/ionex/{ionex_path.parent}")

    # Check if the file is present on the server
    if ionex_path.name not in ftp_connection.nlst():
        log.error(
            f"Failed to download IONEX file {ionex_path.name}: "
            "File not found on the server"
        )
        exit(1)

    # Download the file [TODO: Maybe check response?]
    output_directory.mkdir(parents=True, exist_ok=True)
    _ = ftp_connection.retrbinary(
        f"RETR {ionex_path.name}", output_file.open("wb").write
    )

    return output_file


def decompress_ionex_file(compressed_file: Path) -> Path:
    """Decompress IONEX file

    :param compressed_file_path: Path to the compressed IONEX file
    :return: Path to the decompressed IONEX file
    """

    # Path to the decompressed file
    output_file = compressed_file.with_suffix("")

    # Skip if the decompressed file already exists
    if output_file.exists():
        return output_file

    # Decompress file
    match compressed_file.suffix:

        case ".Z":
            output_file.write_bytes(unlzw3.unlzw(compressed_file.read_bytes()))
        case ".gz":
            with gzip.open(compressed_file, "rb") as f_in:
                output_file.write_bytes(f_in.read())
        case _:
            log.error(
                f"Failed to decompress IONEX file {compressed_file.name}: "
                f"Unexpected extension {compressed_file.suffix}"
            )
            exit(1)

    # Remove compressed file and return
    compressed_file.unlink()
    return output_file


def download_ionex_file_for_date(
    date: "time.Time", output_directory: str | Path
) -> Path:
    """Download and decompress IONEX file for a given date

    The function finds the path to the IONEX file for the given date in the FTP server, downloads it to the specified directory, and decompresses it. The directory is created if it doesn't exist.

    :param date: Date in UTC. Must be at 00:00:00 UTC.
    :param output_directory: Directory to save the downloaded file
    :return ionex_file: Path to the decompressed IONEX file
    """

    # Get path to the IONEX file on the FTP server
    ftp_path = get_ionex_path_for_date(date)

    # Download the compressed IONEX file
    compressed_file = download_compressed_ionex_file(ftp_path, output_directory)

    # If the file is not compressed, return it
    if compressed_file.suffix not in [".Z", ".gz"]:
        return compressed_file

    # Decompress the IONEX file
    ionex_file = decompress_ionex_file(compressed_file)

    return ionex_file
