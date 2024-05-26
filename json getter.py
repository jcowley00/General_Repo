import json
import os
import matplotlib.pyplot as plt

def plot_data_from_dict(data_dict):
    # Extracting relevant data
    name = data_dict.get('name')
    data = data_dict.get('data')

    # Extracting x and y values from data
    x_values = [str(point[0]) for point in reversed(data)]
    y_values = [point[1] for point in reversed(data)]

    # Adding hyphen to x-values formatted as "yyyymm"
    x_values = [value[:4] + '-' + value[4:] if len(value) == 6 else value for value in x_values]

    # Creating the plot
    plt.plot(x_values, y_values)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(name)

    # Setting x-axis ticks
    num_ticks = min(6, len(x_values))  # Limiting to 6 ticks or less
    step = len(x_values) // num_ticks
    plt.xticks(range(0, len(x_values), step), x_values[::step])

    # Saving the plot as a PDF
    plt.savefig(f"{name}.pdf")
    plt.show()



def decode_json_lines(filename):
    with open(filename, 'r') as file:
        current_json = ''
        for line in file:
            current_json += line.strip()
            try:
                decoded_json = json.loads(current_json)
                yield decoded_json
                current_json = ''
            except json.JSONDecodeError:
                pass  # Incomplete JSON, continue reading the file

# Example usage:
os.chdir(r'C:\Users\jbcme\Downloads\NG')
filename = 'NG.txt'



json_dict = {}
counter = 0

for json_obj in decode_json_lines(filename):
    json_dict[json_obj['name']] = json_obj
    plot_data_from_dict(json_obj)
    print(counter)  # Do whatever you need with the decoded JSON object
    counter = counter + 1
