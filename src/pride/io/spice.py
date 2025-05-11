from .resources import load_catalog
from typing import Any
from ..logger import log
from pathlib import Path
import requests

ESA_SPICE = "https://spiftp.esac.esa.int/data/SPICE"
NASA_SPICE = "https://naif.jpl.nasa.gov/pub/naif"


def get_target_information(name: str) -> dict[str, Any]:
    """Get target information from spacecraft catalog

    :param name: Name of the target
    :return: Dictionary containing target information
    """

    name = name.upper()
    catalog = load_catalog("spacecraft.yaml")
    out: dict[str, Any] | None = None

    for target in catalog.values():
        if name in target["names"]:
            out = target
            break
    if out is None:
        log.error(f"Target {name} not found in spacecraft catalog")
        exit(1)

    return out


class SpiceKernelManager:
    """Download and check for presence of SPICE kernels"""

    def __init__(self, mission: str, kernels_folder: Path) -> None:

        # Load mission metadata based on its name
        self.__mission_metadata = get_target_information(mission)
        self.target: str = self.__mission_metadata["short_name"]
        self.__naif_directory: str = self.__mission_metadata["names"][-1]
        self.__metakernel: str = self.__mission_metadata["meta_kernel"]

        # Select the SPICE server for the mission
        __kernel_provider: str = ESA_SPICE
        if self.__mission_metadata["api"] == "NASA":
            __kernel_provider = NASA_SPICE
        self.__api: str = f"{__kernel_provider}/{self.__naif_directory}/kernels"

        # Define path to directory for mission kernels
        self.kernels_folder = kernels_folder / self.target
        if not self.kernels_folder.exists():
            log.info(
                f"Created directory for kernels of {self.target} at "
                f"{self.kernels_folder}"
            )
            self.kernels_folder.mkdir(parents=True)

        return None

    def ensure_metakernel(self) -> Path:
        """Ensures that the metakernel of the mission is present

        :return: Path to the metakernel
        """

        # Path to the metakernel
        metak_path = self.kernels_folder / "metak.tm"
        if metak_path.exists():
            # TODO: Examine to ensure contents are valid
            log.debug(f"Found metakernel for {self.target} at {metak_path}")
            return metak_path

        # Setup for downloading the metakernel
        metakernel_url: str = f"{self.__api}/mk"
        log.info(f"Downloading metakernel for {self.target}")

        # Try to download using the default filename
        response = requests.get(f"{metakernel_url}/{self.__metakernel}")
        if not response.ok:
            # Try to make the filename upercase
            response = requests.get(
                f"{metakernel_url}/{self.__metakernel.upper()}"
            )
            if not response.ok:
                log.error(
                    f"Failed to download metakernel for {self.target}: "
                    f"{response.reason}"
                )
                exit(1)

        # Read contents of the metakernel
        metakernel_content = response.content.decode("utf-8").splitlines()
        assert isinstance(metakernel_content, list)  # Sanity

        # Update metakernel with path to the kernels folder
        with metak_path.open("w") as out:

            for line in metakernel_content:
                # Replace the path to the kernels folder
                if "PATH_VALUES" in line:
                    line = line.replace("..", str(self.kernels_folder))

                # Write the line to the file
                out.write(line + "\n")

        return metak_path

    def __get_required_kernels(self, metakernel: Path) -> list[str]:
        """Get a list with the kernels listed in the metakernel file

        :param metakernel: Path to the metakernel file
        :return required_kernels: List of required kernels
        """

        # Read the metakernel file
        required_kernels: list[str] = []
        metakernel_content = iter(
            [line.strip() for line in metakernel.read_text().splitlines()]
        )

        for current_line in metakernel_content:

            if "KERNELS_TO_LOAD" in current_line:

                # Next line contains data
                current_line = next(metakernel_content)

                # Keep reading lines until end of section: )
                while ")" not in current_line:

                    # If line has content, add it to the list
                    if len(current_line.strip()) > 0:
                        required_kernels.append(
                            current_line.replace("$KERNELS/", "")[1:-1]
                        )

                    # Read next line
                    current_line = next(metakernel_content)

                # If line with delimiter is not empty, add kernel to the list
                __content = current_line.split(")")[0].strip()
                if __content != "":
                    required_kernels.append(
                        __content.replace("$KERNELS/", "")[1:-1]
                    )

                # TODO: What if there are multiple sections?
                break

        return required_kernels

    def ensure_kernels(self, metakernel: Path) -> None:
        """Ensure that the required kernels are present

        :param metakernel: Path to the metakernel file
        """

        # Read required kernels from the metakernel
        required_kernels = self.__get_required_kernels(metakernel)

        # Check if the kernels are present
        for kernel in required_kernels:

            # Path to the kernel
            kernel_path = self.kernels_folder / kernel

            # If already present, skip
            if kernel_path.exists():
                continue

            # Try to download the kernel
            kernel_url = f"{self.__api}/{kernel}"
            log.info(f"Downloading: {kernel}")
            response = requests.get(kernel_url)
            if not response.ok:
                log.error(
                    f"Failed to download {kernel} from {kernel_url}: "
                    f"{response.reason}"
                )
                exit(1)

            # Write kernel to file
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_bytes(response.content)

        return None
