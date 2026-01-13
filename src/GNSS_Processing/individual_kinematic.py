from GNSS_Processing.ppp2mat import save_mat
from GNSS_Processing.PRIDE_plots import plot_GNSS
from data import gps_data_path

from pathlib import Path


def _get_files_and_apply_function(directory_path, function, pattern="*.kin"):
    # Find files matching a pattern
    for file_path in directory_path.glob(pattern):
        if file_path.is_file():
            print(f"Found file matching pattern: {file_path}")
            function(file_path)
            print("Done processing.\n")


def process_kinematic_files(individual_kinematic_dir, downsample=1):
    directory_path = Path(individual_kinematic_dir)

    def process_file(file_path):
        output_path = file_path.with_suffix(".mat")
        print(f"Processing file: {file_path}")
        save_mat(file_path, output_path)
        print(f"Saved processed data to: {output_path}")
        plot_GNSS(output_path, save=True, show=False, downsample=downsample)

    _get_files_and_apply_function(directory_path, process_file)


if __name__ == "__main__":
    individual_kinematic_dir = gps_data_path(
        "GPS_Data/Puerto_Rico/4_PortAft/individual"
    )
    process_kinematic_files(individual_kinematic_dir, downsample=5)
