import json
import numpy as np
import os

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def list_all_items(data):
    for item in data:
        print(item)
        
def find_closest_proposal(reference_time, proposal_data):
    # Calculate midpoint of the reference timestamp
    reference_midpoint = (reference_time[0] + reference_time[1]) / 2
    
    # Initialize variables to store the closest proposal and its difference
    closest_proposal = None
    closest_difference = float('inf')
    
    # Iterate through each proposal
    for proposal in proposal_data:
        # Calculate midpoint of the proposal timestamp
        proposal_midpoint = (proposal['timestamp'][0] + proposal['timestamp'][1]) / 2
        
        # Compute absolute difference between reference and proposal midpoints
        difference = abs(reference_midpoint - proposal_midpoint)
        
        # Update closest proposal if the difference is smaller
        if difference < closest_difference:
            closest_difference = difference
            closest_proposal = proposal
    
    return closest_proposal


def generate_template(scenario_index, segment_numbers):
    template = {}
    template[scenario_index] = []
    for segment_number in segment_numbers:
        template[scenario_index].append({
            "labels": [str(segment_number)],
            "caption_pedestrian": "",
            "caption_vehicle": ""
        })
    return template

segment_numbers = [0, 1, 2, 3, 4]
def format_proposals(scenario_index, proposal_data, reference_data):
    formatted_proposals = generate_template(scenario_index, segment_numbers)
    for idx, reference_time in enumerate(reference_data['timestamps']):
        closest_proposal = find_closest_proposal(reference_time, proposal_data)
        formatted_proposals[scenario_index][idx]["caption_vehicle"] = closest_proposal['sentence']
    return formatted_proposals


def fill_proposals(formatted_proposals, scenario_index, proposal_data, reference_data):
    # Iterate through each reference timestamp
    for idx, reference_time in enumerate(reference_data['timestamps']):
        # Find closest proposal for this reference timestamp
        closest_proposal = find_closest_proposal(reference_time, proposal_data)
        formatted_proposals[scenario_index][idx]["caption_pedestrian"] = closest_proposal['sentence']
    return formatted_proposals


# Provide the path to your JSON file

#####################################################################################################################################

file_path_bdd_veh = 'save/bdd_veh_eval_pdvcl_v_2024-03-25-18-19-20/prediction/num375_epoch0.json' # vehicle-BDD
file_path_bdd_ped = 'save/bdd_ped_eval_pdvcl_v_2024-03-25-18-25-52/prediction/num375_epoch0.json' # pedestrian-BDD
reference_path_bdd = 'data/WTS_DATASET_PUBLIC_TEST/eval_bdd_view.json'

file_path_wts_veh_event = 'save/wts_veh_event_eval_pdvcl_v_2024-03-25-20-53-48/prediction/num48_epoch0.json' # vehicle-WTS-event
file_path_wts_veh_event = 'save/wts_ped_event_eval_pdvcl_v_2024-03-25-20-51-53/prediction/num48_epoch0.json' # pedestrian-WTS-event
reference_path_event = 'data/WTS_DATASET_PUBLIC_TEST/eval_vehicle_view_event.json'

file_path_wts_veh_normal = 'save/wts_veh_normal_eval_pdvcl_v_2024-03-25-20-48-20/prediction/num30_epoch0.json' # vehicle-WTS-normal
file_path_wts_ped_normal = 'save/wts_ped_normal_eval_pdvcl_v_2024-03-25-20-50-14/prediction/num30_epoch0.json' # pedestrian-WTS-normal
reference_path_normal = 'data/WTS_DATASET_PUBLIC_TEST/eval_vehicle_view_normal.json'


file_path_list = [[file_path_bdd_veh, file_path_bdd_ped, reference_path_bdd],[file_path_wts_veh_event, file_path_wts_veh_event, reference_path_event], [file_path_wts_veh_normal, file_path_wts_ped_normal, reference_path_normal]]


folder_path = "data/WTS_DATASET_PUBLIC_TEST/submission_captions"
os.makedirs(folder_path, exist_ok=True)

submission_file = os.path.join(folder_path, 'Team219-Submission-Draft.json')
submission_file_postprocessed = os.path.join(folder_path, 'Team219-Submission-Final.json')

# Read the JSON file

for file_path in file_path_list:
    reference_path = file_path[2]
    veh_file_path = file_path[0]
    ped_file_path = file_path[1]


    reference_data = read_json_file(reference_path)


    #############################################################################################################
    

    
    json_data_veh = read_json_file(veh_file_path)
    proposal_data = json_data_veh['results']
    for data_id in reference_data:
        proposals = proposal_data[data_id]
        references = reference_data[data_id]
        formatted_proposals = format_proposals( data_id, proposals, references)
        
        filename = os.path.join(folder_path, data_id + "_caption.json")
        with open(filename, "w") as f:
            json.dump(formatted_proposals, f, indent=4)
    
    
    
    json_data_ped = read_json_file(ped_file_path)
    proposal_data = json_data_ped['results']
    for data_id in reference_data:
        proposals = proposal_data[data_id]
        references = reference_data[data_id]
        
        filename = os.path.join(folder_path, data_id + "_caption.json")
        # Check if the file exists
        if os.path.exists(filename):
            # Load data from JSON file
            with open(filename, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
                    data = None
     
        formatted_proposals = fill_proposals(data, data_id, proposals, references)
        
        with open(filename, "w") as f:
            json.dump(formatted_proposals, f, indent=4)

print("=====> Results Aggregration Completed. ")

def remove_vehicle_view_from_keys(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Modify keys to remove "_vehicle_view"
    modified_data = {key.replace("_vehicle_view", ""): value for key, value in data.items()}

    # Write modified data back to the JSON file
    with open(json_file, "w") as file:
        json.dump(modified_data, file, indent=4)

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is a JSON file
        if file.endswith('.json'):
            json_file = os.path.join(root, file)
            # Remove "_vehicle_view" from keys
            remove_vehicle_view_from_keys(json_file)
            
print("=====> String Cleaning Finished. ")

# Initialize an empty dictionary to store the merged data
merged_data = {}

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is a JSON file
        if file.endswith('.json'):
            json_file = os.path.join(root, file)
            # Read the contents of the JSON file and merge it into the dictionary
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    # Merge the data into the merged_data dictionary
                    merged_data.update(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {json_file}: {e}")


# Write the merged data to a new JSON file
with open(submission_file, "w") as f:
    json.dump(merged_data, f, indent=4)

print("=====> Submission Draft Completed. ")


# Read the JSON file
with open(submission_file) as file:
    data = json.load(file)

replacements = {
    " UNK h ": " km/h ",
    "km h ": " km/h ",
    " km h ": " km/h ",
    " t UNK ": " T-shirt ",
    " t shirt ": " T-shirt ",
    " closely UNK ": " closely watching ", 
    " pedestrian is about to UNK the road ":" pedestrian is about to cross the road ",
    " slightly UNK speed ":" slightly faster speed ",
    " UNK to the vehicle ":" faced to the vehicle ",
    " UNK ": " "
}

# Modify the text as needed
for key, value in data.items():
    if isinstance(value, list):
        for item in value:
            
            for sub_key, sub_value in item.items():                   
                if isinstance(sub_value, str):
                    # Replace consecutive words with new words
                    for old_word, new_word in replacements.items():
                        item[sub_key] = item[sub_key].replace(old_word, new_word)
                        

# Write the updated data back to the JSON file
with open(submission_file_postprocessed, "w") as file:
    json.dump(data, file, indent=4)

print("=====> Submission File Ready. ")























