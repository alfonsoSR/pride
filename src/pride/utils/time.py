from astropy import time
from ..logger import log

GPS_WEEK_REFERENCE = time.Time("1980-01-06T00:00:00", scale="utc")


def epoch_is_date(epoch: "time.Time") -> bool:
    """Check if the epoch differs from a date by less than 0.01 seconds

    Checks if the time part of the ISOT representation of the epoch is 00:00:00.00, meaning that anything differing from a date by less than 0.01 seconds is considered a date.

    :param epoch: Epoch as a Time object
    :return: True if the epoch is a date, False otherwise
    """

    epoch_isot_str: str = epoch.isot  # type: ignore
    return epoch_isot_str.split("T")[1][:11] == "00:00:00.00"


def get_date_from_epoch(epoch: "time.Time") -> "time.Time":
    """Time object for 00:00:00 UTC of the given epoch

    :param epoch: Epoch in UTC
    :return: Time object for 00:00:00 UTC of the given epoch
    """

    date: "time.Time" = time.Time(epoch.mjd // 1, format="mjd", scale="utc")  # type: ignore
    date.format = "iso"

    return date


def get_gps_week_for_date(date: "time.Time") -> int:
    """Get GPS week for a given date

    The GPS week is the number of weeks since 1980-01-06 00:00:00 UTC.

    :param date: Date in UTC
    :return: GPS week for the given date
    """

    # Ensure that the input is a date, not date and time
    if not epoch_is_date(date):
        log.error(
            f"Failed to get GPS week for {date.isot}: "
            "Date should be at 00:00:00 UTC"
        )
        exit(1)

    # Get GPS week for the given date
    gps_week = int((date - GPS_WEEK_REFERENCE).to("week").value)  # type: ignore
    return gps_week


def get_year_from_epoch(epoch: "time.Time") -> int:
    """Get year from epoch

    :param epoch: Epoch in UTC
    :return: Year component of the epoch
    """

    year: int = int(epoch.datetime.year)  # type: ignore
    return year


def get_hour_from_epoch(epoch: "time.Time") -> int:
    """Get hour from epoch

    :param epoch: Epoch in UTC
    :return: Hour component of the epoch
    """

    hour: int = int(epoch.datetime.hour)  # type: ignore
    return hour


def get_day_of_year_from_epoch(epoch: "time.Time") -> int:
    """Get day of year from epoch

    :param epoch: Epoch in UTC
    :return: Day of year
    """

    day_of_year: int = int(epoch.datetime.timetuple().tm_yday)  # type: ignore
    return day_of_year
