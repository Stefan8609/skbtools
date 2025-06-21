from skbtools.plotting.print_table import printTable


def test_print_table_simple(capsys):
    headers = ["A", "B"]
    data = [(1, 2), (3, 4)]
    table = printTable(headers, data)
    captured = capsys.readouterr()
    assert "A" in table and "B" in table
    assert captured.out.strip() == table.strip()
