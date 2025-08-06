import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from parent directory
from utils import data_utils

# Constants
BASE_DIR = '/home/nirit/IBMProject'
MODEL_NAME_BASE = 'gemini-2.0-flash-001'
concepts_path = os.path.join(BASE_DIR, 'code', 'output_concepts')
output_path = os.path.join(BASE_DIR,'code', 'statistics_output')
def load_dataset(dataset_path, general_concepts, specific_concepts, domain='food', verbose=True):
    """
    Load a dataset from the specified path.
    
    Args:
        dataset_path (str): The path to the dataset file.
        general_concepts (list): A list of general concepts to keep.
        specific_concepts (list): A list of specific concepts to keep.
    Returns:

    """
    n_specific = 10
    dataset =  data_utils.load_json(dataset_path)

    #filter Concepts:
    for item in dataset:
        if general_concepts != 'all':
            item['General Concepts llm as judge'] = {k:v for k,v in item['General Concepts llm as judge'].items() if k in general_concepts}
        if specific_concepts != 'all':
            item['Specific Concepts llm as judge'] = {k:v for k,v in item['Specific Concepts llm as judge'].items() if k in specific_concepts}

    if verbose:
        print(f"Loaded dataset with {len(dataset)} items successfully.")
        print(f"General Concepts: {general_concepts}")
        print(f"Specific Concepts: {specific_concepts}")

    # Change the label of the concepts to -1
    for item in dataset:
        if item['preference'] == -1:
            item['preference'] = 3  # Change -1 to 3 as per the convention
        for key in item['General Concepts llm as judge']:
            if int(item['General Concepts llm as judge'][key]) ==  -1:
                item['General Concepts llm as judge'][key] = 3
        for key in item['Specific Concepts llm as judge']:
            if int(item['Specific Concepts llm as judge'][key]) == -1:
                item['Specific Concepts llm as judge'][key] = 3
    for item in dataset:
        for key in item['General Concepts llm as judge']:
            if int(item['General Concepts llm as judge'][key]) == -1:
                item['General Concepts llm as judge'][key] = 3
        for key in item['Specific Concepts llm as judge']:
            if int(item['Specific Concepts llm as judge'][key]) == -1:
                item['Specific Concepts llm as judge'][key] = 3

    #The concepts labels are saved so 1 represent the chosen response and 3 the rejected, I want to align it so 1 represenr response1 and 3 respponse2
    #in the case where preference is 1, the response1 is the chosen one and response2 is the rejected one, and there is nothing to align
    #in the case where preference is 3, the response1 is the rejected one and response2 is the chosen one, so I need to swap the labels
    for item in dataset:
        if item['preference'] == 3:
            # Swap the labels for general concepts
            for key in item['General Concepts llm as judge']:
                if int(item['General Concepts llm as judge'][key]) == 1:
                    item['General Concepts llm as judge'][key] = 3
                elif int(item['General Concepts llm as judge'][key]) == 3:
                    item['General Concepts llm as judge'][key] = 1
            # Swap the labels for specific concepts
            for key in item['Specific Concepts llm as judge']:
                if int(item['Specific Concepts llm as judge'][key]) == 1:
                    item['Specific Concepts llm as judge'][key] = 3
                elif int(item['Specific Concepts llm as judge'][key]) == 3:
                    item['Specific Concepts llm as judge'][key] = 1
            
    return dataset

def preferance_statistics(dataset, verbose=True):
    """
    Compute preference statistics for the dataset.
    
    Args:
        dataset (list): A list of dataset items.
    Returns:
        dict: A dictionary containing the computed statistics.
    """
    #extract 'preference' from dataset
    preferences = [item['preference'] for item in dataset]
    #count the preferences
    preference_counts = {pref: preferences.count(pref) for pref in set(preferences)}
    #if prefereance is -1 change to 3 du to changing on convention I did during my work:
    if -1 in preference_counts:
        preference_counts[3] = preference_counts.pop(-1)
    #preference can be only 1, '0' or '3', delete other keys and warn if they exist
    valid_preferences = {1, 0, 3}
    for pref in list(preference_counts.keys()):
        if verbose and (pref not in valid_preferences):
            print(f"Warning: Invalid preference '{pref}' found in dataset, count: {preference_counts[pref]}. Removing it.")
            del preference_counts[pref]


    return preference_counts

def concepts_statistics(dataset, general_concepts, specific_concepts, conditioned_statistics = False, verbose=True):
    """
    Compute concepts statistics for the dataset.

    Args:
        dataset (list): A list of dataset items.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    valid_labels = {0, 1, 2, 3}
    # I want to calculate each label probability from the dataset: P(Ci = k), k={0,1,2,3}
    number_of_items = len(dataset)
    # create a dictionary to hold the counts of each label for each concept
    concept_counts = {
        'General Concepts llm as judge': {concept: {label: 0 for label in valid_labels} for concept in general_concepts},
        'Specific Concepts llm as judge': {concept: {label: 0 for label in valid_labels} for concept in specific_concepts}
    }
    for item in dataset:
        # Check if the item has the required keys
        if 'General Concepts llm as judge' not in item or 'Specific Concepts llm as judge' not in item:
            print("Warning: Item is missing 'General Concepts llm as judge' or 'Specific Concepts llm as judge'. Skipping this item.")
            continue
        
        # Validate general concepts
        for concept, label in item['General Concepts llm as judge'].items():
            label = int(label)  # Ensure label is an integer
            if label not in valid_labels:
                print(f"Warning: Invalid label '{label}' for general concept '{concept}'. Expected labels are {valid_labels}.")
            if concept in general_concepts:
                if verbose and concept not in concept_counts['General Concepts llm as judge']:
                    print(f"Warning: Concept '{concept}' not found in general concepts list. Skipping this concept.")
                    continue
                concept_counts['General Concepts llm as judge'][concept][label] += 1
        
        # Validate specific concepts
        for concept, label in item['Specific Concepts llm as judge'].items():
            label = int(label)  # Ensure label is an integer
            if label not in valid_labels:
                print(f"Warning: Invalid label '{label}' for specific concept '{concept}'. Expected labels are {valid_labels}.")
            if concept in specific_concepts:
                if verbose and concept not in concept_counts['Specific Concepts llm as judge']:
                    print(f"Warning: Concept '{concept}' not found in specific concepts list. Skipping this concept.")
                    continue
                concept_counts['Specific Concepts llm as judge'][concept][label] += 1

    # Convert counts to probabilities
    concept_probabilities = {
        'General Concepts llm as judge': {concept: {label: count / number_of_items for label, count in labels.items()} 
                                          for concept, labels in concept_counts['General Concepts llm as judge'].items()},
        'Specific Concepts llm as judge': {concept: {label: count / number_of_items for label, count in labels.items()} 
                                           for concept, labels in concept_counts['Specific Concepts llm as judge'].items()}
    }   
    # Print the computed probabilities
    if verbose:
        print("Computed Concept Probabilities:")
        for concept_type, concepts in concept_probabilities.items():
            print(f"{concept_type}:")
            for concept, labels in concepts.items():
                print(f"  {concept}: {labels}")

    if conditioned_statistics:
        valid_preferences = {1, 0, 3}
        # If conditioned statistics is required, we will filter the dataset based on existing preferences
        # and recursively call this function for each preference.
        existing_preference = set()
        for item in dataset:
            if 'preference' in item:
                existing_preference.add(item['preference'])
        #if ther is preference -1 convert to 3, because I changed the convention during my work
        if -1 in existing_preference:
            existing_preference.remove(-1)
            existing_preference.add(3)
        # Filter out invalid preferences
        existing_preference = [pref for pref in existing_preference if pref in valid_preferences]
        #create a dictionary to hold the conditioned probabilities for each preference
        concept__conditioned_probabilities = {pref: {'General Concepts llm as judge': {}, 'Specific Concepts llm as judge': {}} for pref in existing_preference}
        # Iterate over each preference
        for pref in existing_preference:
            #build dataset with only items with this preference
            filtered_dataset = [item for item in dataset if item['preference'] == pref]
            # Calculate concept probabilities for this preference by recursively calling this function
            concept__conditioned_probabilities[pref], _ = concepts_statistics(filtered_dataset, general_concepts, specific_concepts, conditioned_statistics=False)
    return concept_probabilities, concept__conditioned_probabilities if conditioned_statistics else None

def main(dataset_path, general_concepts, specific_concepts):
    """
    Main function to execute the script.
    """
    dataset = load_dataset(dataset_path, general_concepts, specific_concepts)
    preference_counts = preferance_statistics(dataset, verbose = False)
    concept_probabilities, concept__conditioned_probabilities = concepts_statistics(dataset, general_concepts, specific_concepts, conditioned_statistics=True, verbose=False)
    
    # Print the preference counts
    print("Preference Counts:", preference_counts)
    print("Concept Probabilities:", concept_probabilities)
if __name__ == "__main__":
    domain = 'food'
    #dataset_path = '/home/nirit/IBMProject/code/out_symmetric_data/Modelgemini-2.0-flash-001_Domain_food_FullInfo_symetric_postprocessing_train.txt'  # Path to the dataset
    dataset_path = '/home/nirit/IBMProject/code/output_concepts/Modelgemini-2.0-flash-001_Domain_food_FullInfo_train_number_of_exampels_10.json'
    specific_concepts = data_utils.load_json(os.path.join(concepts_path, f'Model{MODEL_NAME_BASE}_specific_concepts_dict_Domain_{domain}_Test_number_of_exampels_{10}_itertion{10}.json'))
    general_concepts = data_utils.load_json(os.path.join(concepts_path, 'Modelgemini-2.0-flash-001_general_concepts_dict_prev_number_of_exampels_5.json'))
    #general_concepts = 'all'#['Directness','Practicality','Completeness','Clarity','Understanding']  # Example general concepts
    #specific_concepts = 'all'#['IntentAddressed', 'ReliableGuidance']  # Example specific concepts
    main(dataset_path, general_concepts, specific_concepts)