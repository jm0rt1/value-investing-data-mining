# work in progress
from pathlib import Path
import csv
import math

inputdatafilepath = Path(
    ".\scripts\GrahamNumberCalculator\Data\Input\gncDatafile.csv")  # define input file path
outputdatafilepath = Path(
    ".\scripts\GrahamNumberCalculator\Data\Output\gnc-outputDatafile.csv")  # define output file path

data = []  # declare data


def read_data_from_csv(data):
    with open(inputdatafilepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            data.append(row[0:4])  # store the first four columns of each row
            # print(data)
        for row in data:
            print(row)
        return data

    # Close the CSV file
    file.close()


def write_data_to_csv(data):
    # Open the CSV file in write mode
    with open(outputdatafilepath, 'w', newline='') as csv_file:

        # Create a csv.writer object to write to the CSV file
        writer = csv.writer(csv_file)

        # Iterate over the list and write each element to the CSV file
        for row in data:
            writer.writerow(row)
    # Close the CSV file
    csv_file.close()


def graham_number():
    multiplier = 22.5  # define multiplier value for Graham Number Calculation
    read_data_from_csv(data)  # call csv read function

    for row in data:  # iterate through all the rows of csv file to capture eps & bvps
        earn_value_per_share = float(row[2])  # store eps in a variable
        book_value = float(row[3])  # store bvps in a variable
        # print(earn_value_per_share)
        # print(book_value)

        graham_number = round(
            math.sqrt(multiplier * earn_value_per_share * book_value), 2)  # calculate graham number using the formulae
        # print(graham_number)
        # append calculated graham number value to each row
        row.append(graham_number)
    # print(data)
    # call function to wtite the input stock values and graham number to output csv file
    write_data_to_csv(data)


graham_number()
