import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import threading
import concurrent.futures
import matplotlib.ticker as ticker
import numpy as np
from textwrap import wrap
import argparse
import gc




def plot_data_from_dict(data_dict):
    # Extracting relevant data
    name = data_dict.get('name')
    data = data_dict.get('data')

    # Extracting x and y values from data
    x_values = [str(point[0]) for point in reversed(data)]
    y_values = [point[1] for point in reversed(data)]

    # Adding hyphen to x-values formatted as "yyyymm"
    if(len(x_values[0]) == 6):
        x_values = [value[:4] + '-' + value[4:] if len(value) == 6 else value for value in x_values]
    elif(len(x_values[0]) == 8):
        x_values = [value[:4] + '-' + value[4:6]+ '-' + value[6:] if len(value) == 8 else value for value in x_values]

    
    x_numeric = []
    y_numeric = []
    for i in range(len(y_values)):
        if isinstance(y_values[i], (int, float)):
            y_numeric.append(float(y_values[i]))
            x_numeric.append(x_values[i])
        else:
            y_numeric.append(np.nan)
            x_numeric.append(x_values[i])
            
                
    

   # plt.suptitle("\n".join(wrap(plot_name + " (" + unitsshort +") " , 60)))

    # Creating the plot
    plt.plot(x_numeric, y_numeric)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title("\n".join(wrap(name + " (" + data_dict['units'] +") "  , 60)))
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.gca().get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
)
    

    # Setting x-axis ticks
    num_ticks = min(6, len(x_numeric))  # Limiting to 6 ticks or less
    step = len(x_numeric) // num_ticks
    plt.xticks(range(0, len(x_numeric), step), x_numeric[::step],rotation='vertical')

    # Saving the plot as a PDF
    plt.savefig(f"{name}.pdf")
    #plt.show()
    plt.close('all')


def parse_dates(date_strings):
    """
    Parse date strings to convert them to pandas datetime objects.
    Maps "the first", "the second", etc. to consecutive days starting from January 1, 1970.
    
    Parameters:
        date_strings (list): List of date strings.
    
    Returns:
        pd.DatetimeIndex: Pandas datetime index.
    """
    
    try:
        # Custom mapping between strings and Unix timestamps
        unix_timestamps = [i for i in range(len(date_strings))]
        
        # Convert integers to corresponding Unix timestamps (number of days since January 1, 1970)
        unix_timestamps = [(pd.to_datetime('1970-01-01') + pd.Timedelta(days=i)).timestamp() for i in unix_timestamps]
        
        # Convert Unix timestamps to pandas datetime objects
        return pd.to_datetime(unix_timestamps, unit='s'), unix_timestamps
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return None, None

def seasonal_decompose_and_plot(data, pd_in, plot_name,unitsshort):
    """
    Perform seasonal decomposition on a time series represented by a list of lists,
    and plot the original time series along with its decomposed components.
    
    Parameters:
        data (list): List of lists where each inner list contains [date_string, value].
        period (int): The periodicity of the seasonal component.
        plot_name (str): Name of the plot and the output PDF file.
    """
    period_num = 0
    if(pd_in == 'A'):
        period_num = 1
    elif(pd_in == 'M'):
        period_num = 12
    elif(pd_in == 'W'):
        period_num = 52
    else:
        period_num = 365
    
    # Extract dates and values from data
    dates = [item[0] for item in reversed(data)]
    if(len(dates[0]) == 6):
        dates = [value[:4] + '-' + value[4:] if len(value) == 6 else value for value in dates]
    elif(len(dates[0]) == 8):
        dates = [value[:4] + '-' + value[4:6]+ '-' + value[6:] if len(value) == 8 else value for value in dates]

    values = [item[1] for item in reversed(data)]

    
    # Parse dates
    dates_index, integers = parse_dates(dates)
    
    if dates_index is None:
        return  # Exit if date parsing fails
    
    # Create a pandas Series with the time index and values
    ts = pd.Series(values, index=dates_index)
    
    # Perform seasonal decomposition
    result = seasonal_decompose(ts, model='additive', period=period_num)
    
    # Plot the original time series and its components
    plt.figure(figsize=(12, 10))
    # Plot title
    
    #plt.suptitle(plot_name + " (" + unitsshort +") " , fontsize=16)
    plt.suptitle("\n".join(wrap(plot_name + " (" + unitsshort +") " , 60)))
    # Plot original time series
    plt.subplot(311)
    plt.plot(ts)
    plt.xticks(dates_index, dates, rotation=45)
    plt.title('Original')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(fontsize=5)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.gca().get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
)
    
    # Plot seasonal component
    plt.subplot(312)
    plt.plot(result.seasonal)
    plt.xticks(dates_index, dates, rotation=45)
    plt.title('Seasonal')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(fontsize=5)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.gca().get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
)
    
    # Plot sum of trend and residual components
    plt.subplot(313)
    plt.plot(result.trend + result.resid)
    plt.xticks(dates_index, dates, rotation=45)
    plt.title('Trend + Residual')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(fontsize=5)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.gca().get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
)
    
    # Save plot as PDF
    plt.savefig(plot_name + ".pdf")
    
    #plt.show()
    plt.close('all')


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


def manager_func(json_obj):
    try_other = False
    

    try:
        seasonal_decompose_and_plot(json_obj['data'],json_obj['f'],json_obj['name'],json_obj['units'])
    except:
        try_other = True
    
    if(try_other):
        try:
            plot_data_from_dict(json_obj)
        except:
            print(json_obj['name'])
            
    gc.collect()
            
def main(start_idx,end_idx):            
    # Example usage:
    os.chdir(r'C:\Users\jbcme\Downloads\PET')
    filename = 'PET.txt'
    
    
    json_list = []
    
    for json_obj in decode_json_lines(filename):
        json_list.append(json_obj)
    
        
    os.chdir(r'C:\Users\jbcme\Downloads\EBA\pdfs')
    
    # Define your list of 450 things
    
    
    # Define the number of threads you want to use    
    json_list = json_list[start_idx:end_idx]    
    counter = 0
    for i in json_list:
        if(i['name'] == 'Gulf Coast (PADD 3) Blender Net Input of Biodiesel/Renewable Diesel Fuel, Monthly'):
            manager_func(i)
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process some numbers.')
#     parser.add_argument('argument1', type=str, help='First argument to the script')
#     parser.add_argument('argument2', type=str, help='Second argument to the script')
#     args = parser.parse_args()
#     main(int(args.argument1), int(args.argument2))

main(8000,12000)
