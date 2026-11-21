"""Splice consecutive hourly RINEX 2.11 observation files into one file.

The Bermuda receivers logged one RINEX 2.11 file per hour (sbp2rinex 2.4.3). Solving
each hour as its own PRIDE session restarts the ambiguities every hour and puts
decimetre steps in the positions, so the hours are joined before PPP. The header of
the first file is kept, with TIME OF LAST OBS from the last file and one COMMENT
line added; epoch records are copied unchanged. Inputs must share the observation
types and must not overlap in time.
"""

from pathlib import Path


def split_header(path):
    """Return (header lines, body lines) of a RINEX 2 observation file."""
    lines = Path(path).read_text().splitlines(keepends=True)
    end = next((i for i, ln in enumerate(lines) if "END OF HEADER" in ln[60:]), None)
    if end is None:
        raise ValueError(f"{path}: no END OF HEADER line")
    if not lines[0][:9].strip().startswith("2"):
        raise ValueError(f"{path}: expected RINEX 2.x, got version {lines[0][:9]!r}")
    return lines[: end + 1], lines[end + 1 :]


def obs_types(header):
    """Observation types from the '# / TYPES OF OBSERV' records."""
    types = []
    for ln in header:
        if "# / TYPES OF OBSERV" in ln[60:]:
            types += ln[6:60].split()
    return types


def epoch_times(body, n_types):
    """(yy, mm, dd, hh, mi, ss) of every observation epoch in a RINEX 2 body.

    Walks the records using the satellite count of each epoch line, so a
    misaligned body raises instead of being silently misread.
    """
    lines_per_sat = 1 + (n_types - 1) // 5
    epochs = []
    i = 0
    while i < len(body):
        ln = body[i]
        flag = ln[28:29].strip()
        n_sat = int(ln[29:32])
        if flag in ("", "0", "1"):
            epochs.append(
                tuple(int(ln[k : k + 3]) for k in (0, 3, 6, 9, 12))
                + (float(ln[15:26]),)
            )
            # Satellite list continues on extra lines beyond 12 satellites.
            i += 1 + (n_sat - 1) // 12 + n_sat * lines_per_sat
        else:
            # Event flags 2-5: n_sat is the number of special records that follow.
            i += 1 + n_sat
    return epochs


def splice_rinex2(input_paths, output_path):
    """Write the RINEX 2 files in input_paths (in time order) as one file."""
    input_paths = [Path(p) for p in input_paths]
    headers, bodies = zip(*(split_header(p) for p in input_paths), strict=True)
    types = obs_types(headers[0])
    epochs = []
    for path, header, body in zip(input_paths, headers, bodies, strict=True):
        if obs_types(header) != types:
            raise ValueError(f"{path}: observation types differ from {input_paths[0]}")
        file_epochs = epoch_times(body, len(types))
        if not file_epochs:
            raise ValueError(f"{path}: no observation epochs")
        if epochs and file_epochs[0] <= epochs[-1]:
            raise ValueError(
                f"{path}: starts at {file_epochs[0]}, not after {epochs[-1]}"
            )
        epochs += file_epochs

    last_obs = next(ln for ln in headers[-1] if "TIME OF LAST OBS" in ln[60:])
    header = [last_obs if "TIME OF LAST OBS" in ln[60:] else ln for ln in headers[0]]
    first, last = input_paths[0].name, input_paths[-1].name
    comment = f"spliced {len(input_paths)} files {first}..{last}"
    header.insert(len(header) - 1, f"{comment:<60.60}COMMENT\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.writelines(header)
        for body in bodies:
            f.writelines(body)

    _, out_body = split_header(output_path)
    if epoch_times(out_body, len(types)) != epochs:
        raise RuntimeError(f"{output_path}: written epochs differ from the inputs")
    print(
        f"{output_path.name}: {len(input_paths)} files, {len(epochs)} epochs, "
        f"{epochs[0]} -> {epochs[-1]}"
    )
    return output_path


if __name__ == "__main__":
    # DOY 168 only (00:37:10-23:37:09 GPST): 000N-05040 starts on DOY 167, for which
    # no WUM products are in PRIDE-PPPAR/Products.
    hourly_root = Path("/Users/brandt/PRIDE-PPPAR/BIOSTesting")
    output_dir = Path(__file__).resolve().parents[3] / "Bermuda_Data/GPS_Data/RINEX"
    first_slot, last_slot = 5100, 6420

    for antenna in (1, 2, 3, 4):
        obs_dir = hourly_root / f"Antenna{antenna}" / "ObsData"
        files = [
            obs_dir / f"{antenna:04d}-{slot:05d}.obs"
            for slot in range(first_slot, last_slot + 1, 60)
        ]
        splice_rinex2(files, output_dir / f"ber{antenna}1680.20o")
