#!/bin/bash


# # Set the new value
# new_age=35

# for nb in 5
# do
#     new_age=$nb
#     # Use jq to update the JSON file
#     jq '.age = $new_age' --argjson new_age "$new_age" config.json > temp.json && mv temp.json config.json
# done


for bs in  8 4
do
    
    # Use jq to update the JSON file
    jq '.train_batch_size = $bs' --argjson bs "$bs" ds_zero2_config.json > temp.json && mv temp.json ds_zero2_config.json

done


