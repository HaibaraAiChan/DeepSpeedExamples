
def print_vals(val_list):
    # Print the selected values
    for value in val_list:
        print(value)


# Specify the file path
file_path = '0.log'  # Replace with the path to your file
file_path = '0,1.log'  # Replace with the path to your file
# file_path = '0,2.log'  # Replace with the path to your file
# file_path = '0,1,2.log'  # Replace with the path to your file

# Initialize a list to store the values
fw_values = []
bw_values = []
opt_values = []
max_mem=0
# Open the file and read lines starting from line 3918
with open(file_path, 'r') as file:
    lines = file.readlines()[1869:1956]  # Lines are 0-indexed, so 3917 corresponds to line 3918
    # print(lines)
    # Iterate through the lines
    for line in lines:
        # Check if the line contains "fwd_microstep: "
        if "fwd_microstep: " in line:
            # Extract the value after "fwd_microstep: " and add it to the list
            value_list = line.split("|")[1:]
            # print(value_list)
            fw_value = float(value_list[0].split("fwd_microstep: ")[1].strip())
            bw_value = float(value_list[1].split("bwd_microstep: ")[1].strip())
            fw_values.append(fw_value)
            bw_values.append(bw_value)
        if "optimizer_allgather: " in line:
            # Extract the value after "optimizer_allgather: " and add it to the list
            value_list = line.split("|")[1:]
            # print(value_list)
            op1_value = float(value_list[0].split("optimizer_allgather: ")[1].strip())
            op2_value = float(value_list[1].split("optimizer_gradients: ")[1].strip())
            op3_value = float(value_list[2].split("optimizer_step: ")[1].strip())
            # print(op1_value+op2_value+op3_value)
            opt_values.append(op1_value+op2_value+op3_value)
        if "MaxMemAllocated=" in line:
            value_list = line.split(",")[5]
            max_mem = value_list.split("=")[1].strip()
            
        
            
            
# print('forward time')
# print_vals(fw_values)
print("forward time average  (sec) ", sum(fw_values)/len(fw_values))

print()

# print()
# # print("backward time")
# # print_vals(bw_values)
print("backward time average (sec) ", sum(bw_values)/len(bw_values))

print()

print("optimizer time average (sec)", sum(opt_values)/len(opt_values))

print('-'*60)



avg_iteration=(sum(fw_values)+sum(bw_values)+sum(opt_values))/len(fw_values)
print('the avg iteration time (sec)', avg_iteration)
print()

print('-='*40)
print("max cuda mem allocated :",max_mem)
print()