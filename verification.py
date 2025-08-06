


import json
import os


def number_of_jsons_extracted(json_objects, wanted_number = 2):
    if len(json_objects) != wanted_number:
        return False
    else:
        return True
def format_json_objects(json_objects):
    """
    Checks the format of the JSON objects.
    Args:
        json_objects (list): List of JSON objects containing concept data.

    first json file should be in the format:
    {
        item1: {
            domain: "domain1",
            concept1: "concept1",
            concept2: "concept2",
            ...
        },
        item2: {
            domain: "domain2",
            concept1: "concept1",
            concept2: "concept2",
            ...
        },
    }

    second json file should be in the format:
    {
        general_concepts:{
            concept1: "concept1",
            concept2: "concept2",
            ...
        },
        domain_specific_concepts: {
            domain1: {
                concept1: "concept1",
                concept2: "concept2",
                ...
            },       
    }
    """
    first_json = json_objects[0]
    second_json = json_objects[1]
    # Check if the first JSON object is in the correct format
    if not isinstance(first_json, dict):
        return False
    for i, (item, data) in enumerate(first_json.items()):
        key_expected = f"item{i+1}"
        if not isinstance(data, dict):
            return False
        if key_expected != item:
            return False
        if 'domain' not in data:
            return False
        
    #check if the second JSON object is in the correct format
    if not isinstance(second_json, dict):
        return False
    if 'general_concepts' not in second_json:
        return False
    if 'domain_specific_concepts' not in second_json:
        return False
    if not isinstance(second_json['general_concepts'], dict):
        return False
    if not isinstance(second_json['domain_specific_concepts'], dict):
        return False
    
    return True
        
    

def check_number_domain_concepts(json_objects, max_specific_concepts):
    """
    Check if the number of recived domain-specific concepts is as expected."
    This is a proxy to unserstand if I got the same concept for all items in the same domain.
    Args:
        json_objects (list): List of JSON objects containing concept data.
    """
    specific_concepts_recived = 0
    for k,v in json_objects[1]['domain_specific_concepts'].items():
        specific_concepts_recived += len(v)
    if specific_concepts_recived > max_specific_concepts:
        return False
    else:
        return True
def check_same_concepts_within_domains(json_objects, specific_concepts):
    #assuming the last concepts in the list are the domain specific ones
    #And that I can check the last specific_concepts elements in the list, which represent #S I gave the llm since prev specific domain already checked
    concepts_dict = {}
    for item,v in json_objects[0].items():
        current_domain = v['domain']
        current_concepts = list(v.keys())
        #take the last "specific_concept" elements in the list
        current_concepts = current_concepts[-specific_concepts:]
        if current_domain not in concepts_dict: #First otem in the domain, int the list
            concepts_dict[current_domain] = None
            concepts_dict[current_domain] = current_concepts
        else: #not the first item in the domain, check is the concepts are the same
            if current_concepts != concepts_dict[current_domain]:
                print(f"Error: The concepts for the domain '{current_domain}' do not match.")
                return False
            else:
                continue
    return True


    
def all_subdicts_have_same_keys(dictionary):
    # Get the keys of the first sub-dictionary
    first_keys = set(next(iter(dictionary.values())).keys())
    
    # Compare the keys of all other sub-dictionaries
    for sub_dict in dictionary.values():
        if set(sub_dict.keys()) != first_keys:
            return False
    # TO DO : change it in future to go through all domains and finf if there is a couple with the same keys
    return True

def check_number_general_concepts(json_objects, max_general_concepts):
    """
    Check if the number of general concepts is as expected.
    Args:
        json_objects (list): List of JSON objects containing concept data.
    """
    # Assuming the Second JSON object contains the general concepts
    general_concept_recived = len(json_objects[1]['general_concepts'])
    if general_concept_recived > max_general_concepts:
        return False
    else:
        return True
    
    
def run_all_verification(json_objects, wanted_jsons,  max_general_concepts, max_specific_concepts ,specific_concepts, json_format = True, json_number = True, 
                         general_concepts = True,domain_concepts = True, same_concepts = True, all_subdicts = True):
    """
    Run all verifications on the JSON objects.
    Args:
        json_objects (list): List of JSON objects containing concept data.
        specific_concepts (int): Number of domain-specific concepts expected.
    """
    if json_number:
        if not number_of_jsons_extracted(json_objects, wanted_jsons):
            print(f"Error: expected {wanted_jsons} json .")
            return False
    if json_format:
        if not format_json_objects(json_objects):
            print("Error: JSON objects are not in the expected format.")
            return False
    if domain_concepts:
        if not check_number_domain_concepts(json_objects, max_specific_concepts):
            print("Error: Number of domain-specific concepts exceeds the limit.")
            return False
    if general_concepts:
        if not check_number_general_concepts(json_objects, max_general_concepts):
            print("Error: Number of general concepts exceeds the limit.")
            return False
    if same_concepts:
        if not check_same_concepts_within_domains(json_objects, int(specific_concepts)):
            print("Error: Concepts within domains do not match.")
            return False
    if all_subdicts:
        if all_subdicts_have_same_keys(json_objects[1]['domain_specific_concepts']):
            print("Error: sub-dictionaries have the same keys.")
            return False
    # If all verifications pass
    print("All verifications passed.")
    return True
    
