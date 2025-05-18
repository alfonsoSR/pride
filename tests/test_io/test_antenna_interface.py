from pride.io.antenna_parameters.interface import (
    load_raw_parameters_from_catalog,
    turn_raw_parameters_into_tuple,
    AntennaParameters,
)
import pytest

HOBART_PARAMETERS = (
    "HOBART12",
    "FO_SECN",
    "MO_AZEL",
    False,
    "ME_COMP",
    12.0,
    0.0,
    0.0,
    1005.7,
    12.0,
    0.0,
    0.0,
    1e-5,
    6.708,
    1.14e-05,
    0.0021,
    1.2e-05,
    1.8009,
    2.24e-05,
    4.262,
    2.24e-05,
)


@pytest.mark.parametrize(
    ["station_name", "expected_line"],
    [
        (
            "AGGO",
            (
                "ANTENNA_INFO  AGGO      FO_PRIM MO_AZEL RA_NO   ME_COMP  17.0"
                "  0    0   1011.7    6.0  0.2700   4.73  1.00E-5   1.4600 "
                "1.20E-5   0.0000 1.20E-5   0.7500 1.20E-5   4.5000 1.20E-5"
            ),
        ),
        (
            "HOBART12",
            (
                "ANTENNA_INFO  HOBART12  FO_SECN MO_AZEL RA_NO   ME_COMP  12.0"
                "  0    0   1005.7   12.0  0.0000  0      1.00E-5   6.7080 1."
                "14E-5   0.0021 1.20E-5   1.8009 2.24E-5   4.2620 2.24E-5"
            ),
        ),
        ("Invalid", None),
    ],
    ids=["Base case", "Cited in header", "Invalid"],
)
def test_load_raw_parameters_from_catalog(
    station_name: str, expected_line: str | None
) -> None:

    loaded_line = load_raw_parameters_from_catalog(station_name)
    if expected_line is not None:
        assert loaded_line == expected_line
    else:
        assert loaded_line is None

    return None


@pytest.mark.parametrize(
    ["raw_parameters", "expected_tuple"],
    [
        (
            (
                "ANTENNA_INFO  HOBART12  FO_SECN MO_AZEL RA_NO   ME_COMP  12.0"
                "  0    0   1005.7   12.0  0.0000  0      1.00E-5   6.7080 1."
                "14E-5   0.0021 1.20E-5   1.8009 2.24E-5   4.2620 2.24E-5"
            ),
            HOBART_PARAMETERS,
        )
    ],
)
def test_turn_raw_parameters_into_tuple(
    raw_parameters: str, expected_tuple: tuple
) -> None:

    obtained_tuple = turn_raw_parameters_into_tuple(raw_parameters)
    assert obtained_tuple == expected_tuple


@pytest.mark.parametrize(
    ["station_name", "expected_tuple"],
    [("HOBART12", HOBART_PARAMETERS)],
)
def test_data_structure_from_catalog(
    station_name: str, expected_tuple: tuple
) -> None:

    from_catalog = AntennaParameters.from_catalog(station_name)
    from_expected_tuple = AntennaParameters(*expected_tuple)
    assert from_catalog == from_expected_tuple

    # Check all the attributes
    attribute_list = [
        "ivs_name",
        "focus_type",
        "mount_type",
        "radome",
        "meas_type",
        "T0",
        "sin_T",
        "cos_T",
        "h0",
        "ant_diam",
        "hf",
        "df",
        "gamma_hf",
        "hp",
        "gamma_hp",
        "AO",
        "gamma_AO",
        "hv",
        "gamma_hv",
        "hs",
        "gamma_hs",
    ]
    for idx, attribute in enumerate(attribute_list):
        assert getattr(from_catalog, attribute) == expected_tuple[idx]

    return None


def test_data_structure_from_catalog_missing_ok() -> None:

    invalid = AntennaParameters.from_catalog_missing_ok("Invalid")
    assert invalid is None

    valid = AntennaParameters.from_catalog_missing_ok("HOBART12")
    assert isinstance(valid, AntennaParameters)

    return None
