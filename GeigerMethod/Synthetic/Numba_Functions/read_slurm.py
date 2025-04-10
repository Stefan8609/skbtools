import numpy as np
import re


def parse_iteration_line(line):
    # Extract the iteration number
    iter_match = re.search(r'Iteration (\d+)', line)
    if not iter_match:
        return None

    iteration = int(iter_match.group(1))

    # Extract the Lever vector
    lever_match = re.search(r'Lever: \[(.*?)\]', line)
    if not lever_match:
        return None

    lever_str = lever_match.group(1)
    lever = np.array([float(x) for x in lever_str.split()])

    # Extract the Offset
    offset_match = re.search(r'Offset: (\d+\.\d+)', line)
    if not offset_match:
        return None

    offset = float(offset_match.group(1))

    # Extract the RMSE
    rmse_match = re.search(r'RMSE: (\d+\.\d+)', line)
    if not rmse_match:
        return None

    rmse = float(rmse_match.group(1))

    return {
        'iteration': iteration,
        'lever': lever,
        'offset': offset,
        'rmse': rmse
    }


def read_iteration_file(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_iteration_line(line)
            if parsed:
                results.append(parsed)
    return results


# Usage
file_path = "slurm-2395955.out"
iteration_data = read_iteration_file(file_path)

# Filter iterations where z-index of lever is greater than -7
filtered_iterations = [iter_data for iter_data in iteration_data
                      if len(iter_data['lever']) >= 3 and iter_data['lever'][2] < -5]

# Find the iteration with the lowest RMSE among the filtered iterations
if filtered_iterations:
    min_rmse_iter = min(filtered_iterations, key=lambda x: x['rmse'])
    print(f"Best iteration with z-index > -7: {min_rmse_iter['iteration']}")
    print(f"Lever: {min_rmse_iter['lever']}")
    print(f"Offset: {min_rmse_iter['offset']}")
    print(f"RMSE: {min_rmse_iter['rmse']}")
else:
    print("No iterations found where the z-index of the lever is greater than -7.")