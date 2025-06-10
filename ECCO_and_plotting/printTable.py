import random
import string

"""Simple utilities for rendering data tables in the console."""


def printTable(headers, data):
    """Print a formatted table.

    Parameters
    ----------
    headers : list of str
        Labels for each table column.
    data : sequence of tuple
        Rows to display. Each tuple must match the header length.

    Returns
    -------
    str
        The rendered table as a string.
    """

    try:
        headers[len(data[0]) - 1]
    except Exception:
        print("Error: Number of headers does not match size of data")

    # Determine max width of each column
    max_widths = [
        max(len(str(header)), max(len(str(row[i])) for row in data))
        for i, header in enumerate(headers)
    ]

    # Create the table with headers
    table = ""
    for i, header in enumerate(headers):
        table += f"{header:<{max_widths[i]}}  | "
    table = table[:-2]
    table += "\n" + "-" * (sum(max_widths) + len(headers) * 3 - 1) + "\n"

    # Add the data
    for row in data:
        for i, value in enumerate(row):
            table += f"{value:<{max_widths[i]}}  | "
        table = table[:-2]
        table += "\n"

    print(table)
    return table


def demo():
    """Example usage generating random values."""
    headers = ["x", "y", "z", "string"]
    data = []
    for _ in range(10):
        tup = (
            random.random() * 10,
            random.random() * 10,
            random.random() * 10,
            "".join(random.choices(string.ascii_uppercase + string.digits, k=12)),
        )
        data.append(tup)
    printTable(headers, data)


if __name__ == "__main__":
    demo()
