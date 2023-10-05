"""
Function that neatly prints a table given headers and data
Written by Stefan Kildal-Brandt (with assistance from chat GPT)

Inputs:
    headers (list len=n): The headers of each column in the table
    data (list len=n of tuples): The data that you want to be printed (each value
        in the tuple corresponds to the header of the same index)
Outputs:
    table (string): The string that contains the table that we desire to print
"""

def printTable(headers, data):
    try:
        headers[len(data[0])-1]
    except:
        print("Error: Number of headers does not match size of data")

    #Determine max width of each column
    max_widths = [max(len(str(header)), max(len(str(row[i])) for row in data)) for i, header in enumerate(headers)]

    #Create the table with headers
    table = ""
    for i,header in enumerate(headers):
        table += f"{header:<{max_widths[i]}}  | "
    table = table[:-2]
    table += "\n" + "-" * (sum(max_widths) + len(headers) * 3 - 1) + "\n"

    #Add the data
    for row in data:
        for i, value in enumerate(row):
            table += f"{value:<{max_widths[i]}}  | "
        table = table[:-2]
        table += "\n"

    print(table)
    return table

"""
Demo - Demonstrates how a table is printed given headers and data. Prints random numbers and random string
Inputs:
    None
Outputs:
    None
Uncomment below to run
"""
import random
import string

def demo():
    headers = ["x", "y", "z", "string"]
    data = []
    for i in range(10):
        tup = (random.random()*10,random.random()*10,random.random()*10, ''.join(random.choices(string.ascii_uppercase +
                                 string.digits, k=12)))
        data.append(tup)
    printTable(headers, data)
if __name__ == "__main__":
    demo()