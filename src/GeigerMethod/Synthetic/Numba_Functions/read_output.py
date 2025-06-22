import numpy as np
import re


def parse_grid_search_line(line):
    """Parse a line from the grid search output file.

    Parameters
    ----------
    line : str
        Single line from the text output.

    Returns
    -------
    dict or None
        Parsed values or ``None`` if the line is a header.
    """
    # Skip header lines
    if line.startswith("Grid Search Results") or line.startswith("[Lever]"):
        return None

    # Extract lever array
    lever_match = re.search(r"\[(.*?)\]", line)
    if not lever_match:
        return None
    lever_str = lever_match.group(1)
    lever = np.array([float(x) for x in lever_str.split(",")])

    # Extract CDOG estimate
    cdog_match = re.search(r"\[(.*?)\]", line[lever_match.end() :])
    if not cdog_match:
        return None
    cdog_str = cdog_match.group(1)
    cdog = np.array([float(x) for x in cdog_str.split(",")])

    # Extract other values: offset, time_bias, esv_bias, rmse
    remaining = line[lever_match.end() + cdog_match.end() :]
    values = [
        float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", remaining)
    ]

    if len(values) < 4:
        return None

    return {
        "lever": lever,
        "cdog": cdog,
        "offset": values[0],
        "time_bias": values[1],
        "esv_bias": values[2],
        "rmse": values[3],
    }


def read_grid_search_file(filename):
    """Load an output file produced by ``grid_search``.

    Parameters
    ----------
    filename : str
        Path to the text file.

    Returns
    -------
    list of dict
        Parsed entries for each iteration.
    """
    results = []
    with open(filename, "r") as f:
        for line in f:
            parsed = parse_grid_search_line(line)
            if parsed:
                results.append(parsed)
    return results


def find_best_iterations(results, z_threshold=-7):
    """Return the iteration with minimum RMSE below a depth threshold."""
    # Filter iterations where z-index of lever is greater than threshold
    filtered_results = [
        result for result in results if result["lever"][2] < z_threshold
    ]
    # Find the iteration with the lowest RMSE among the filtered iterations
    if filtered_results:
        min_rmse_iter = min(filtered_results, key=lambda x: x["rmse"])
        return min_rmse_iter
    else:
        return None


# Usage example
if __name__ == "__main__":
    CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    z_threshold = 0

    file_path = "output.txt"  # Replace with your actual file path
    results = read_grid_search_file(file_path)

    # Find the best iteration with lever z-index greater than -7
    best_iter = find_best_iterations(results, z_threshold=z_threshold)

    if best_iter:
        print(f"Best iteration with z-index < {z_threshold}:")
        print(f"Lever: {best_iter['lever']}")
        print(f"CDOG Estimate: {best_iter['cdog'] - CDOG_guess_base}")
        print(f"Offset: {best_iter['offset']}")
        print(f"Time Bias: {best_iter['time_bias']}")
        print(f"ESV Bias: {best_iter['esv_bias']}")
        print(f"RMSE: {best_iter['rmse']}")
    else:
        print("No iterations found where the z-index of the lever is greater than -7.")
    # Additional statistics
    print(f"\nTotal iterations analyzed: {len(results)}")
    print(f"Average RMSE: {np.mean([r['rmse'] for r in results]):.6f}")
    print(f"Min RMSE: {np.min([r['rmse'] for r in results]):.6f}")
    print(f"Max RMSE: {np.max([r['rmse'] for r in results]):.6f}")
