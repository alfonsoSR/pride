from pride import io
from pathlib import Path
from matplotlib import pyplot as plt

# Code of the station to be examined and output directory
station = "Cd"
output_directory = Path("output-gr035").resolve()

if __name__ == "__main__":

    # Read data from binary file of station
    source = output_directory / f"GR035_{station}.del"
    _, data = io.DelFile(source).read()

    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    mjd_00 = data[0].mjd_ref + (data[0].mjd2[0] / 86400)

    for idx, scan in enumerate(data):

        t = ((scan.mjd_ref + (scan.mjd2 / 86400)) - mjd_00) * 24.0
        ax.plot(t, scan.delays, ".")

    ax.set_xlabel("Hours past initial epoch")
    ax.set_ylabel("Delay [s]")

    plt.show()
