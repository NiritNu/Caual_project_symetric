import random
import json

import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file) 


def save_text(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def remove_duplicates(items, key):
    seen = set()
    count_duplicates = 0
    unique_items = []
    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            unique_items.append(item)
        else:
            count_duplicates += 1
    print(f"Removed {count_duplicates} duplicates based on key '{key}'.")
    return unique_items 

def extract_item_by_original_index(items, original_index):
    """
    Extracts an item from a list of dictionaries based on the 'original_index' key.

    Args:
        items (dict): dictonat by domain, each value is a list of dictionaries.
        original_index (int): The value of the 'original_index' to search for.

    Returns:
        dict: The first dictionary that matches the 'original_index', or None if not found.
    """
    original_index = int(original_index)  # Ensure original_index is an integer
    for domain, data_domain in items.items():
        for item in data_domain:
            if item.get('original_index') == original_index:
                return item
    print(f"No item found with original_index: {original_index}")
    return None

def balance_data_per_batch(items, statistics):
    number_of_non_zero = statistics['y=1'] + statistics['y=3']
    nuber_of_one_wanted = int(np.ceil(number_of_non_zero / 2))
    nuber_of_three_wanted = number_of_non_zero - nuber_of_one_wanted
    if nuber_of_one_wanted < statistics['y=1']: #I need to comvert some y=1 to y=3
        for item in items:
            if item['preference'] == 1:
                tmp = item['response2']
                item['response2'] = item['response1']
                item['response1'] = tmp
                item['preference'] = 3
                statistics['y=3'] += 1
                statistics['y=1'] -= 1
                if nuber_of_one_wanted == statistics['y=1']:
                    return items, statistics
    elif nuber_of_three_wanted < statistics['y=3']: #I need to comvert 3 to 1
        for item in items:
            if item['preference'] == 3:
                tmp = item['response2']
                item['response2'] = item['response1']
                item['response1'] = tmp
                item['preference'] = 1
                statistics['y=1'] += 1
                statistics['y=3'] -= 1
                if nuber_of_three_wanted == statistics['y=3']:
                    return items, statistics
    else: # the items are already balanced
        return items, statistics
                
def balamce_data_per_domain(items, verbose=True):
    #calculate domain statistics
    domain_statistics = {}
    for domain, data_domain in items.items():
        domain_statistics[domain] = {'y=1': 0, 'y=3': 0, 'y=0': 0}
        for item in data_domain:
            if item['preference'] == 1:
                domain_statistics[domain]['y=1'] += 1
            elif item['preference'] == 3:
                domain_statistics[domain]['y=3'] += 1
            elif item['preference'] == 0:
                domain_statistics[domain]['y=0'] += 1
        #balance data per domain
        items[domain], domain_statistics[domain] = balance_data_per_batch(items[domain], domain_statistics[domain])
    statistics = {'y=1': 0, 'y=3': 0, 'y=0': 0}
    for domain, stats in domain_statistics.items():
        statistics['y=1'] += stats['y=1']
        statistics['y=3'] += stats['y=3']
        statistics['y=0'] += stats['y=0']
    if verbose:
        print("Domain statistics after balancing:")
        for domain, stats in domain_statistics.items():
            print(f"{domain}: {stats}")
        print(f"Total statistics: {statistics}")
    return items, statistics

def remove_duplicates_debug_concepts(items, key):
    seen = set()
    count_duplicates = 0
    unique_items = []
    for data_domain in items.values():
        for item in data_domain.values():
            value = item.get(key)
            if value not in seen:
                seen.add(value)
                unique_items.append(item)
            else:
                count_duplicates += 1
    
    print(f"Removed {count_duplicates} duplicates based on key '{key}'.")
    return unique_items 

def train_test_overlap_check(train_data, test_data):
    """
    Checks if there is any overlap between the training and test data.

    Args:
        train_data (dict): The training data dictionary.
        test_data (dict): The test data dictionary.

    Returns:
        bool: True if there is overlap, False otherwise.
    """
    train_set = set()
    for domain in train_data.keys():
        if isinstance(train_data[domain], list):
            # If train_data is a list, we assume each item is a dictionary with 'User Query' key
            for item in train_data[domain]:
                train_set.add(item['original_index'])
        elif isinstance(train_data[domain], dict):
            for item in train_data[domain].values():
                train_set.add(item['User Query'])

    for domain in test_data.keys():
        if isinstance(test_data[domain], list):
            # If test_data is a list, we assume each item is a dictionary with 'User Query' key
            for item in test_data[domain]:
                if item['original_index'] in train_set:
                    return True
        elif isinstance(test_data[domain], dict):
            for item in test_data[domain].values():
                if item['User Query'] in train_set:
                    return True
    return False
    
    

def replace_double_quotes(text): #This function is used , so the json file will be valid
  """
  Replaces all double quotes (") in a given text with single quotes (').

  Args:
    text: The input string.

  Returns:
    A new string with all double quotes replaced by single quotes.
  """
  return text.replace('"', "'") 
 
def creating_data_dict(domains):
    data_dict = {}
    for i in range(len(domains[0])):
        raw_data = load_json(domains[0][i])
        tmp_dict = {}
        for key in raw_data.keys():
            Query = replace_double_quotes(raw_data[key]['user_query'])
            Chosen = replace_double_quotes(raw_data[key]['chosen'])
            Rejected = replace_double_quotes(raw_data[key]['rejected'])
            tmp_dict[key] = {'User Query':Query, 'Chosen Response':Chosen, 'Rejected Response':Rejected,
                              'Domain':domains[1][i], 'Index':raw_data[key]['index'], 'Original_index':raw_data[key]['original_index']}
        data_dict[domains[1][i]] = tmp_dict
    return data_dict

def sample_data(data_dict, n,previuos_sampeld_index = []):
    # Sample n examples from each domain
    sampled_examples = []
    print(len(previuos_sampeld_index))
    if len(previuos_sampeld_index) > 0:
        # If there are already sampled keys, remove them from the data_dict
        for domain in data_dict.keys():
            for key in list(data_dict[domain].keys()):
                inner_dict = data_dict[domain][key]
                if inner_dict['Original_index'] in previuos_sampeld_index:
                    del data_dict[domain][key]
    for domain in data_dict.keys():
        sampled_keys = random.sample(list(data_dict[domain].keys()), n)
        #save original_index of the sampled keys
        for key in sampled_keys:
            previuos_sampeld_index.append(data_dict[domain][key]['Original_index']) 
        for key in sampled_keys:
            sampled_examples.append(data_dict[domain][key])
    return sampled_examples, previuos_sampeld_index

def split_data(data_dict, train_ratio=0.75):
    """
    Splits the data into training, and test sets.

    Args:
        data_dict (dict): The data dictionary containing the data to be split.
        each key is a domain and each value is a dictionary of examples.
        train_ratio (float): The ratio of the data to be used for training.

    Returns:
        Two dictonaries: train_data and test_data.
        each dictonary contains keys as domain names and values as a list of examples.
    """
    train_data = {}
    test_data = {}
    for domain in data_dict.keys():
        examples = list(data_dict[domain].items())
        random.shuffle(examples)
        split_index = int(len(examples) * train_ratio)
        train_data[domain] = dict(examples[:split_index])
        test_data[domain] = dict(examples[split_index:])
    return train_data, test_data


def sample_data_by_order(data_dict, n, slice = 0):
    """
    Samples n examples from each domain in the order they appear in the data_dict.

    Args:
        data_dict (dict): The data dictionary containing the data to be sampled.
        n (int): The number of examples to sample from each domain.
        slice(int): the current position of the data_dict to start sampling from.
        If slice is 0, it will sample from the beginning of the data_dict.
        If slice is 1, it will sample from the second item in the first dict of the data_dict, and so own and so forth.

    Returns:
        list: A list of sampled examples. (list of dicts)
    """
    sampled_examples = []
    #create a list of all the keys in the data_dict
    examples = []
    for domain in data_dict.keys():
        examples += data_dict[domain]
    #sample n examples from the examples list, by the current slice
    sampled_list = examples[slice:n+slice]
    return sampled_list, n+slice

