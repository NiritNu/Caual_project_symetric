import os
import json

def save_change_dict(new_data_path,General_concept_dict_0_minus1,General_concept_dict_0_1,General_concept_dict_0_2,
                                                                       General_concept_dict_1_minus1,General_concept_dict_1_0,General_concept_dict_1_2,
                                                                       General_concept_dict_2_minus1,General_concept_dict_2_0,General_concept_dict_2_1,
                                                                       Food_concept_dict_0_minus1,Food_concept_dict_0_1,Food_concept_dict_0_2,
                                                                       Food_concept_dict_1_minus1,Food_concept_dict_1_0,Food_concept_dict_1_2,
                                                                       Food_concept_dict_2_minus1,Food_concept_dict_2_0,Food_concept_dict_2_1,
                                                                       Legal_concept_dict_0_minus1,Legal_concept_dict_0_1,Legal_concept_dict_0_2,
                                                                       Legal_concept_dict_1_minus1,Legal_concept_dict_1_0,Legal_concept_dict_1_2,
                                                                       Legal_concept_dict_2_minus1,Legal_concept_dict_2_0,Legal_concept_dict_2_1, d, iter_number, trails, agrresive,key,domain,model,batch_size = None):
    #save all concepts dict
    if not agrresive:
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_0_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_0_2, f, indent=4)    
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_1_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_General_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(General_concept_dict_2_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_0_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_0_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_1_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Food_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Food_concept_dict_2_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_1_2, f, indent=4)  
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Model_{model}_Key_{key}_Domain_{domain}_Legal_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_1, f, indent=4)
        
    else:
        with open(os.path.join(new_data_path, f'General_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_0_1, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_0_2, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_1_2, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'General_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(General_concept_dict_2_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_0_1, f, indent=4)       
        with open(os.path.join(new_data_path, f'Food_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_0_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_1_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Food_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Food_concept_dict_2_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_0_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_0_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_1, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_0_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_0_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_1_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_1_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_1_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_1_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_1_2_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:  
            json.dump(Legal_concept_dict_1_2, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_2_minus1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_minus1, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_2_0_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_0, f, indent=4)
        with open(os.path.join(new_data_path, f'Legal_concept_dict_2_1_{d}_itreation_{iter_number}_trail_{trails}_aggresive.json'), 'w') as f:
            json.dump(Legal_concept_dict_2_1, f, indent=4)


def dict_change_extraction(General_concepts, d,original_label, new_label, key,General_concept_dict_0_minus1,General_concept_dict_0_1,General_concept_dict_0_2,
                                                                       General_concept_dict_1_minus1,General_concept_dict_1_0,General_concept_dict_1_2,
                                                                       General_concept_dict_2_minus1,General_concept_dict_2_0,General_concept_dict_2_1,
                                                                       Food_concept_dict_0_minus1,Food_concept_dict_0_1,Food_concept_dict_0_2,
                                                                       Food_concept_dict_1_minus1,Food_concept_dict_1_0,Food_concept_dict_1_2,
                                                                       Food_concept_dict_2_minus1,Food_concept_dict_2_0,Food_concept_dict_2_1,
                                                                       Legal_concept_dict_0_minus1,Legal_concept_dict_0_1,Legal_concept_dict_0_2,
                                                                       Legal_concept_dict_1_minus1,Legal_concept_dict_1_0,Legal_concept_dict_1_2,
                                                                       Legal_concept_dict_2_minus1,Legal_concept_dict_2_0,Legal_concept_dict_2_1):

    if key in General_concepts.keys():
        if (original_label ==0) & ( new_label == 3):
            relevent_dict = General_concept_dict_0_minus1
        elif (original_label ==0) & ( new_label == 1):
            relevent_dict = General_concept_dict_0_1
        elif (original_label ==0) & ( new_label == 2):
            relevent_dict = General_concept_dict_0_2
        elif (original_label ==1) & ( new_label == 3):
            relevent_dict = General_concept_dict_1_minus1
        elif (original_label ==1) & ( new_label == 0):
            relevent_dict = General_concept_dict_1_0
        elif (original_label ==1) & ( new_label == 2):
            relevent_dict = General_concept_dict_1_2
        elif (original_label ==2) & ( new_label == 3):
            relevent_dict = General_concept_dict_2_minus1
        elif (original_label ==2) & ( new_label == 0):
            relevent_dict = General_concept_dict_2_0
        elif (original_label ==2) & ( new_label == 1):
            relevent_dict = General_concept_dict_2_1
    else:
        if  (original_label ==0) & ( new_label == 3):
            relevent_dict = Food_concept_dict_0_minus1 if d == 'food' else Legal_concept_dict_0_minus1
        elif (original_label ==0) & ( new_label == 1):
            relevent_dict = Food_concept_dict_0_1 if d == 'food' else Legal_concept_dict_0_1
        elif (original_label ==0) & ( new_label == 2):
            relevent_dict = Food_concept_dict_0_2 if d == 'food' else Legal_concept_dict_0_2
        elif (original_label ==1) & ( new_label == 3):
            relevent_dict = Food_concept_dict_1_minus1 if d == 'food' else Legal_concept_dict_1_minus1
        elif (original_label ==1) & ( new_label == 0):
            relevent_dict = Food_concept_dict_1_0 if d == 'food' else Legal_concept_dict_1_0
        elif (original_label ==1) & ( new_label == 2):
            relevent_dict = Food_concept_dict_1_2 if d == 'food' else Legal_concept_dict_1_2
        elif (original_label ==2) & ( new_label == 3):
            relevent_dict = Food_concept_dict_2_minus1 if d == 'food' else Legal_concept_dict_2_minus1
        elif (original_label ==2) & ( new_label == 0):
            relevent_dict = Food_concept_dict_2_0 if d == 'food' else Legal_concept_dict_2_0
        elif (original_label ==2) & ( new_label == 1):
            relevent_dict = Food_concept_dict_2_1 if d == 'food' else Legal_concept_dict_2_1
    return relevent_dict

def update_normalizatopn_factors(new_label, old_label,General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2,
    Food_concept_norm_0_2, Legal_concept_norm_0_2,
    General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1,
    General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0,
    General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2,
    General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1,
    General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0,
    General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,
    General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1):
    
    if new_label == 1:
        if old_label == 0:
            General_concept_norm_0_1 = General_concept_norm_0_1 + 1
            Food_concept_norm_0_1 = Food_concept_norm_0_1 + 1
            Legal_concept_norm_0_1 = Legal_concept_norm_0_1 + 1
        elif old_label == 2:
            General_concept_norm_2_1 = General_concept_norm_2_1 + 1
            Food_concept_norm_2_1 = Food_concept_norm_2_1 + 1
            Legal_concept_norm_2_1 = Legal_concept_norm_2_1 + 1
    elif new_label == 2:
        if old_label == 0:
            General_concept_norm_0_2 = General_concept_norm_0_2 + 1
            Food_concept_norm_0_2 = Food_concept_norm_0_2 + 1
            Legal_concept_norm_0_2 = Legal_concept_norm_0_2 + 1
        elif old_label == 1:
            General_concept_norm_1_2 = General_concept_norm_1_2 + 1
            Food_concept_norm_1_2 = Food_concept_norm_1_2 + 1
            Legal_concept_norm_1_2 = Legal_concept_norm_1_2 + 1
    elif new_label == 3:
        if old_label == 0:
            General_concept_norm_0_minus1 = General_concept_norm_0_minus1 + 1
            Food_concept_norm_0_minus1 = Food_concept_norm_0_minus1 + 1
            Legal_concept_norm_0_minus1 = Legal_concept_norm_0_minus1 + 1
        elif old_label == 1:
            General_concept_norm_1_minus1 = General_concept_norm_1_minus1 + 1
            Food_concept_norm_1_minus1 = Food_concept_norm_1_minus1 + 1
            Legal_concept_norm_1_minus1 = Legal_concept_norm_1_minus1 + 1
        elif old_label == 2:
            General_concept_norm_2_minus1 = General_concept_norm_2_minus1 + 1
            Food_concept_norm_2_minus1 = Food_concept_norm_2_minus1 + 1
            Legal_concept_norm_2_minus1 = Legal_concept_norm_2_minus1 + 1 
    elif new_label == 0:
        if old_label == 1:
            General_concept_norm_1_0 = General_concept_norm_1_0 + 1
            Food_concept_norm_1_0 = Food_concept_norm_1_0 + 1
            Legal_concept_norm_1_0 = Legal_concept_norm_1_0 + 1
        elif old_label == 2:
            General_concept_norm_2_0 = General_concept_norm_2_0 + 1
            Food_concept_norm_2_0 = Food_concept_norm_2_0 + 1
            Legal_concept_norm_2_0 = Legal_concept_norm_2_0 + 1   

    return General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
           General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
           General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
           General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
           General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1