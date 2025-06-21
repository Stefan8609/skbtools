import numpy as np
import re


def parse_grid_search_results(filename):
    """Parse the best entry from a grid search output file."""
    min_rmse = float("inf")
    min_lever = None
    min_grid = None

    with open(filename, "r") as file:
        # Skip header lines
        next(file)  # Skip "Grid Search Results"
        next(file)  # Skip column headers

        for line in file:
            # Extract the three components using regex
            match = re.match(r"\[(.*?)\], \[(.*?)\], (\d+\.\d+)", line)
            if match:
                lever_str, grid_str, rmse_str = match.groups()

                # Parse RMSE (easiest part)
                rmse = float(rmse_str)

                # Parse lever arm
                lever = np.array([float(x.strip()) for x in lever_str.split(",")])

                # Parse GPS grid (trickier)
                # Clean up the grid string and convert to a numpy array
                grid_str = grid_str.replace("[", "").replace("]", "")
                grid_values = [float(x.strip()) for x in grid_str.split() if x.strip()]
                grid = np.array(grid_values).reshape(4, 3)

                # Check if this is the minimum RMSE so far
                if rmse < min_rmse:
                    min_rmse = rmse
                    min_lever = lever
                    min_grid = grid

    return min_lever, min_grid, min_rmse


if __name__ == "__main__":
    filename = "gaussian_output.txt"  # Change this to your actual filename

    min_lever, min_grid, min_rmse = parse_grid_search_results(filename)

    print("Results with lowest RMSE:")
    print(f"RMSE: {min_rmse:.4f}")
    print(f"Lever arm: {min_lever}")
    print("GPS Grid Estimate:")
    print(min_grid)
