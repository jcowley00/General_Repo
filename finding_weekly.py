import pandas as pd
import numpy as np
import numba
from numba import prange,jit,njit
import chart_maker
import os
import PyPDF2


def concatenate_pdfs(pdf_list, output_path):
    # Create a PDF merger object
    merger = PyPDF2.PdfMerger()
    
    # Loop through the list of PDFs and append each one to the merger
    for pdf in pdf_list:
        merger.append(pdf)
    
    # Write the concatenated PDF to the output path
    merger.write(output_path)
    merger.close()


def combine_array_and_vector(array, vector):
    # Ensure the vector is a column vector
    vector = vector.reshape(-1, 1)
    
    # Combine the array and the vector as a new column
    combined_array = np.column_stack((array, vector))
    
    return combined_array


@numba.njit(cache = True)
def array_and_coordinates(array):
    # Flatten the array to get the actual values in a vector
    values = array.flatten()
    
    # Get the shape of the array
    rows, cols = array.shape
    
    # Generate the indices and reshape them to get a vector of coordinates
    row_indices, col_indices = np.indices((rows, cols))
    coordinates = np.column_stack((row_indices.flatten(), col_indices.flatten()))
    
    # Stack the values and coordinates together into a single 2D array with 3 columns
    combined_array = np.column_stack((values, coordinates))
    
    return combined_array




@numba.njit(cache = True)
def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Compute the slope (m) and intercept (b)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept

@numba.njit(cache = True)
def directional_acc (array_1, array_2, slope, intercept,sigma_move):
    counter = 0
    denominator = 0
    input_array_std = np.std(array_1)    
    upper_bound = (-(intercept/slope)) + (sigma_move * input_array_std)
    lower_bound = (-(intercept/slope)) - (sigma_move * input_array_std)
    temp_array_1 = np.zeros((len(array_2)), dtype=float)
    temp_array_2 = np.zeros((len(array_2)), dtype=float)
    
    wrong_guess_error_array_1 = np.zeros((len(array_2)), dtype=float)
    wrong_guess_error_array_2 = np.zeros((len(array_2)), dtype=float)
    wrong_guess_error_counter = 0
    
    
    for i in range(len(array_1)):
        if(array_1[i] > upper_bound or array_1[i] < lower_bound):
            temp_array_1[denominator] = array_1[i]
            temp_array_2[denominator] = array_2[i]
            
            denominator = denominator + 1
            if(intercept + (array_1[i]*slope) > 0):
                if(array_2[i] > 0):
                    counter = counter + 1
                else:
                    
                    #time of error
                    wrong_guess_error_array_1[wrong_guess_error_counter] = i
                    #actual error
                    wrong_guess_error_array_2[wrong_guess_error_counter] = abs(((array_1[i] * slope) + intercept) - array_2[i])
                    wrong_guess_error_counter = wrong_guess_error_counter + 1
                    
                    
            
            else:
                if(array_2[i] < 0):
                    counter = counter + 1
                else:
                    #time of error
                    wrong_guess_error_array_1[wrong_guess_error_counter] = i
                    #actual error
                    wrong_guess_error_array_2[wrong_guess_error_counter] = abs(((array_1[i] * slope) + intercept) - array_2[i])
                    wrong_guess_error_counter = wrong_guess_error_counter + 1

                    
        
    if(denominator == 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return max(counter / denominator,1 - counter / denominator), np.corrcoef(temp_array_1[0:denominator],temp_array_2[0:denominator])[1,0],np.corrcoef(wrong_guess_error_array_1[0:wrong_guess_error_counter],wrong_guess_error_array_2[0:wrong_guess_error_counter])[1,0],denominator,upper_bound,lower_bound
            


#return_matrix[i,k] = np.corrcoef(input_matrix[first_matrix_start:first_matrix_end,i],input_matrix[second_matrix_start:second_matrix_end,k])[1,0]
            
    
@numba.njit(parallel = True,cache = True)
def direction_matrix_maker (directional_matrix,slope_matrix,intercept_matrix, general_correl_matrix,reduced_correl_matrix,reduced_error_correl_matrix,outside_sigma_count,upper_bound,lower_bound,input_matrix,first_matrix_start,first_matrix_end,second_matrix_start,second_matrix_end,sigma_move):
    
    for i in prange(len(input_matrix[0])):
        for k in range(len(input_matrix[0])):
            slope, intercept = linear_regression(input_matrix[first_matrix_start:first_matrix_end,i],input_matrix[second_matrix_start:second_matrix_end,k])
            directional_matrix[i,k], reduced_correl_matrix[i,k],reduced_error_correl_matrix[i,k],outside_sigma_count[i,k],upper_bound[i,k],lower_bound[i,k] = directional_acc(input_matrix[first_matrix_start:first_matrix_end,i],input_matrix[second_matrix_start:second_matrix_end,k],slope,intercept,sigma_move)
            slope_matrix[i,k] = slope
            intercept_matrix[i,k] = intercept
            general_correl_matrix[i,k]= np.corrcoef(input_matrix[first_matrix_start:first_matrix_end,i],input_matrix[second_matrix_start:second_matrix_end,k])[1,0]
            

week_load = pd.read_csv('weekly_out.csv')
week_load = week_load.drop(columns=['Unique Week'])
numpy_frame = week_load.to_numpy()

all_stocks = list(week_load.columns)

directional_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
slope_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
intercept_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)

#correlation for all returns
general_correl_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
#correlation for returns outside of n std devs
reduced_correl_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
#correlation for returns outside of n std devs
reduced_error_correl_matrix = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)

outside_sigma_count = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
upper_bound = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)
lower_bound = np.zeros((len(all_stocks), len(all_stocks)), dtype=float)


first_matrix_start = 154
first_matrix_end = 205

second_matrix_start = 155
second_matrix_end = 206

#correl_matrix_maker(matrix_zeros,numpy_frame,first_matrix_start, first_matrix_end,second_matrix_start,second_matrix_end)
direction_matrix_maker(directional_matrix,slope_matrix,intercept_matrix,general_correl_matrix,reduced_correl_matrix,reduced_error_correl_matrix,outside_sigma_count,upper_bound,lower_bound,numpy_frame,first_matrix_start, first_matrix_end,second_matrix_start,second_matrix_end,.5)





######################################################################################
directional_array = np.abs(directional_matrix)
values = array_and_coordinates(np.nan_to_num(directional_array, nan=0.0))

sigma_values = array_and_coordinates(np.nan_to_num(outside_sigma_count, nan=0.0))

values = combine_array_and_vector(values,sigma_values[:,0])
t_f_array = values[:,3] >= 20
values = values[t_f_array]
sorted_indices = np.argsort(values[:, 0])
sorted_array = values[sorted_indices][::-1]
sorted_array = sorted_array[:200]


os.chdir(r'C:\Users\jbcme\Python Files\Correl Side finder\pdfs')
for i in sorted_array:
    #print(week_load.columns[i[0]] + week_load.columns[i[1]])
    chart_maker.write_chart(numpy_frame[first_matrix_start:first_matrix_end,int(i[1])],numpy_frame[second_matrix_start:second_matrix_end,int(i[2])],week_load.columns[int(i[1])] + " & " + week_load.columns[int(i[2])],upper_bound[int(i[1]),int(i[2])],lower_bound[int(i[1]),int(i[2])])
    
    
    
pdf_list = []   
os.chdir(r'C:\Users\jbcme\Python Files\Correl Side finder\pdfs')
for i in os.listdir():
    pdf_list.append(i)


concatenate_pdfs(pdf_list,"output.pdf")