import math
import os
import sys
import traceback
import random
import json
from langchain_google_vertexai import VertexAI
#from vertexai import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, ChatSession, HarmCategory, HarmBlockThreshold


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
import verification
from utils import data_utils, prompt_utils, json_utils, change_label_utils, building_data

path_to_creds = r"/home/nirit/IBMProject/HowToRunForMNitay/gemma-nlp-422508-c4d4ce542552-gilat.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds

# Constants
BASE_DIR = '/home/nirit/IBMProject'

PARENT_PROMPT_PATH = os.path.join(BASE_DIR, 'code', 'prompts')
BASE_PROMPT_PATH = [os.path.join(PARENT_PROMPT_PATH, 'LLMasAjudgeGraphPhase.txt'), os.path.join(PARENT_PROMPT_PATH, 'GeneralConceptsClassification.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLabel2minusOne.txt'), os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLAbel2minusOneSimpler.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLabel2minusSimplerAgressive.txt')]
concepts_path = os.path.join(BASE_DIR, 'code', 'output_concepts')
output_path = os.path.join(BASE_DIR,'code', 'out_graph_built')
new_data_path = os.path.join(BASE_DIR,'code', 'out_graph_built', 'new_data_to_save')
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(new_data_path):
    os.makedirs(new_data_path)

NAMES_LIST = ['food', 'legal']

MODEL_NAME = "gemini-2.0-flash-001"#'gemini-2.5-flash-preview-05-20'#"gemini-2.0-flash-001" #"gemini-pro" #chevk which is better
MODEL_NAME_BASE = 'gemini-2.0-flash-001' #for the base model, to load the concepts
def set_seed(seed):
    random.seed(seed)


def preprocessing_data(ChangeminusOne):
    '''
    ChangeminusOne - change the sign of label -1 to 3, so the llm will get less confused
    '''
     #Load General concepts (legal + food)
    domains = ['food', 'legal']
    trainset = {'food': [], 'legal': []}
    testset = {'food': [], 'legal': []}
    specific_concepts = {'food': [], 'legal': []}
    n_specific = 10
    for d in domains:
        trainset[d] =  data_utils.load_json(os.path.join(concepts_path,f'Model{MODEL_NAME_BASE}_Domain_{d}_FullInfo_train_number_of_exampels_{n_specific}.json'))
        testset[d] =  data_utils.load_json(os.path.join(concepts_path,f'Model{MODEL_NAME_BASE}_Domain_{d}_FullInfo_test_number_of_exampels_{n_specific}.json'))
        #load specific_cooncepts
        specific_concepts[d] = data_utils.load_json(os.path.join(concepts_path, f'Model{MODEL_NAME_BASE}_specific_concepts_dict_Domain_{d}_Test_number_of_exampels_{10}_itertion{10}.json'))

    General_concepts = data_utils.load_json(os.path.join(concepts_path, 'Modelgemini-2.0-flash-001_general_concepts_dict_prev_number_of_exampels_5.json'))
    print("END LOADING DATA")
    # keep only the concepts that are in the list
    for d in domains:
        for item in trainset[d]:
            # Collect keys to delete from 'General Concepts llm as judge'
            keys_to_delete = [key for key in item['General Concepts llm as judge'] if key not in General_concepts]
            for key in keys_to_delete:
                del item['General Concepts llm as judge'][key]

            # Collect keys to delete from 'Specific Concepts llm as judge'
            keys_to_delete = [key for key in item['Specific Concepts llm as judge'] if key not in specific_concepts[d]]
            for key in keys_to_delete:
                del item['Specific Concepts llm as judge'][key]
        for item in testset[d]:
            # Collect keys to delete from 'General Concepts llm as judge'
            keys_to_delete = [key for key in item['General Concepts llm as judge'] if key not in General_concepts]
            for key in keys_to_delete:
                del item['General Concepts llm as judge'][key]

            # Collect keys to delete from 'Specific Concepts llm as judge'
            keys_to_delete = [key for key in item['Specific Concepts llm as judge'] if key not in specific_concepts[d]]
            for key in keys_to_delete:
                del item['Specific Concepts llm as judge'][key]

    # Change the label of the concepts to -1
    if ChangeminusOne:
        for d in domains:
            for item in trainset[d]:
                for key in item['General Concepts llm as judge']:
                    if int(item['General Concepts llm as judge'][key]) ==  -1:
                        item['General Concepts llm as judge'][key] = 3
                for key in item['Specific Concepts llm as judge']:
                    if int(item['Specific Concepts llm as judge'][key]) == -1:
                        item['Specific Concepts llm as judge'][key] = 3
            for item in testset[d]:
                for key in item['General Concepts llm as judge']:
                    if int(item['General Concepts llm as judge'][key]) == -1:
                        item['General Concepts llm as judge'][key] = 3
                for key in item['Specific Concepts llm as judge']:
                    if int(item['Specific Concepts llm as judge'][key]) == -1:
                        item['Specific Concepts llm as judge'][key] = 3

    # delete from test set utems that are in trainset:
    for d in domains:
        trainset[d] = data_utils.remove_duplicates(trainset[d], 'original_index')
        testset[d] = data_utils.remove_duplicates(testset[d], 'original_index')
        # remove items from test set that are in train set
        #testset[d] = [item for item in testset[d] if item['original_index'] not in [train_item['original_index'] for train_item in trainset[d]]]
    
    #check if there is any overlap between train and test set
    overlap = data_utils.train_test_overlap_check(trainset, testset)
    return trainset, testset, General_concepts, specific_concepts

                         

def main():
    #init seed
    set_seed(24)

    # init model
    #model = generative_models.GenerativeModel(model_name=MODEL_NAME)
    model = GenerativeModel(model_name=MODEL_NAME)
    # Start the chat session BEFORE calling your function
    # This chat_session will maintain the history across multiple calls
    #chat_session = model.start_chat(history=[])
    #model_classification = generative_models.GenerativeModel(model_name=MODEL_NAME)
    model_classification = GenerativeModel(model_name=MODEL_NAME)
    temperature = 0.5
    agressive_change =False
    # Load the data
    trainset, testset, General_concepts, specific_concepts = preprocessing_data(ChangeminusOne = True)
    # create concept list of all the concepts
    concept_list = list(General_concepts.keys()) + list(specific_concepts['food'].keys()) + list(specific_concepts['legal'].keys())
    Filter_concept_flag = True 
    if Filter_concept_flag:
        wanted_concepts_general = ['Directness','Practicality','Completeness','Clarity','Understanding']
        wanted_concepts_specific = {'food': ['IntentAddressed', 'ReliableGuidance',], 'legal': ['LegalRelevance', 'PracticalImplication']}
        #keep in General_concepts only the wanted concepts
        General_concepts = {key: General_concepts[key] for key in wanted_concepts_general if key in General_concepts}
        #keep in specific_concepts only the wanted concepts - uncomment in next runs 
        #specific_concepts = {d: {key: specific_concepts[d][key] for key in wanted_concepts_specific[d] if key in specific_concepts[d]} for d in specific_concepts.keys()}

    all_wanted_concepts = wanted_concepts_general + list(specific_concepts['food'].keys()) + list(specific_concepts['legal'].keys())
    
    # Phase One: change the labels of the concepts so all will be 1 (Represebt thw chosen concept more then the rejected one)
    domains = ['food', 'legal']
    index_for_debug_food = [i for i in range(295)]#+ [i for i in range(100, 110)]
    index_for_debug_legal = [i for i in range(295)]#[i for i in range(60)]
    batch_for_recursion = []
    batch_size = 1
    mismatch = 0
    llm_unnecessry_modification = 0
    
    mismatch_tracking = {'food': [], 'legal': []}
    new_data_to_save = {'food': [], 'legal': []}
    inner_dict = {'0-1': 0, '0-2': 0, '0-3': 0, '1-0': 0, '1-2': 0, '1-3': 0, '2-0': 0, '2-1': 0, '2-3': 0, '3-0': 0, '3-1': 0, '3-2': 0}
    #General_concept_dict_0_minus1 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_0_minus1 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_0_minus1 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_0_minus1 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()}
    #General_concept_dict_0_1 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_0_1 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_0_1 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_0_1 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()} 
    #General_concept_dict_0_2 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_0_2 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_0_2 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_0_2 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()}  
    #General_concept_dict_2_minus1 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_2_minus1 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_2_minus1 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_2_minus1 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()}
    # General_concept_dict_2_1 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_2_1 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_2_1 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_2_1 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()} 
    # General_concept_dict_2_0 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_2_0 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_2_0 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_2_0 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()}
    # General_concept_dict_1_minus1 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_1_minus1 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_1_minus1 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_1_minus1 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()} 
    # General_concept_dict_1_0 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_1_0 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_1_0 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_1_0 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()} 
    # General_concept_dict_1_2 = {key: inner_dict.copy() for key in General_concepts.keys()} 
    General_concept_dict_1_2 = {key: inner_dict.copy() for key in wanted_concepts_general}
    Food_concept_dict_1_2 = {key: inner_dict.copy() for key in specific_concepts['food'].keys()}
    Legal_concept_dict_1_2 = {key: inner_dict.copy() for key in specific_concepts['legal'].keys()}

    # Normalization factor of th dictionries:
    General_concept_norm_0_minus1 = 0
    Food_concept_norm_0_minus1 = 0
    Legal_concept_norm_0_minus1 = 0
    General_concept_norm_0_1 = 0
    Food_concept_norm_0_1 = 0
    Legal_concept_norm_0_1 = 0
    General_concept_norm_0_2 = 0
    Food_concept_norm_0_2 = 0
    Legal_concept_norm_0_2 = 0
    General_concept_norm_1_minus1 = 0
    Food_concept_norm_1_minus1 = 0
    Legal_concept_norm_1_minus1 = 0     
    General_concept_norm_1_0 = 0
    Food_concept_norm_1_0 = 0
    Legal_concept_norm_1_0 = 0
    General_concept_norm_1_2 = 0
    Food_concept_norm_1_2 = 0
    Legal_concept_norm_1_2 = 0
    General_concept_norm_2_minus1 = 0
    Food_concept_norm_2_minus1 = 0
    Legal_concept_norm_2_minus1 = 0
    General_concept_norm_2_0 = 0
    Food_concept_norm_2_0 = 0
    Legal_concept_norm_2_0 = 0
    General_concept_norm_2_1 = 0
    Food_concept_norm_2_1 = 0
    Legal_concept_norm_2_1 = 0
    

    for d in domains:    
        lenght_of_trainset = math.ceil(len(trainset[d]))
        item_count = 0
        error_count = 0
        iter_number = 0
        tot_iterations = lenght_of_trainset // batch_size
        error_in_llm_as_judge_label = 0
        no_change_were_made_even_when_needed = 0
        agree_count = 0
        dis_agree_count = 0
        while iter_number < tot_iterations:
            try:
                # restart conversation for each iteration
                chat_session = model.start_chat(history=[])
                if error_count > 100:
                    print("Error: Too many errors occurred. Exiting.")
                    break
                if d == 'food':
                    if iter_number in index_for_debug_food:
                        debug_current = True
                    else:
                        debug_current = False
                else:
                    if iter_number in index_for_debug_legal:
                        debug_current = True
                    else:
                        debug_current = False
                start_index = iter_number * batch_size
                end_index = min(start_index + batch_size, len(trainset[d]))
                batch = trainset[d][start_index:end_index]
                print(f"\033[1;31m{iter_number}\033[0m")
               
                TargetConcept = wanted_concepts_general[0]
                #TargetConcept = list(General_concepts.keys())[0]
                #TargetConcept = list(specific_concepts[d].keys())[0]
                text = building_data.ChangeMinusOne(batch, General_concepts, specific_concepts[d], TargetConcept,chat_session, temperature,domain = d, n=batch_size, debug=debug_current, iteration=iter_number, trails = 1, agressive=agressive_change)
                json_objects_change_minus1 = json_utils.check_json_extraction(text, 1)
                if json_objects_change_minus1 == None:
                    error_count += 1
                    print("Error: JSON extraction failed. Please check the text.")
                    continue
                    #check classification of the concepts for the modified labels
                json_objects_change_minus1 = json_utils.check_all_items_in_json(json_objects_change_minus1, 1)
                new_batch = []
                for index, item in json_objects_change_minus1[0].items():
                    #find the currect item in the batch:
                    for b in batch:
                        if 'Original Index' in b:
                            batch_index = str(b['Original Index'])
                        elif 'original_index' in b:
                            batch_index = str(b['original_index'])
                        if item['Original Index'] == batch_index:
                            new_batch.append(b)
                            break

                batch = new_batch 
                current_label_for_chat = [] 
                # preferamce calculation:
                preferance = building_data.llmAsJudgeAvreage(json_objects_change_minus1, TargetConcept, model_classification, temperature,batch_size ,debug=debug_current, iteration=iter_number, iteration_end=tot_iterations, test_flag=False, domain=d)  

                if not agressive_change:
                    msn = 'minus_one'
                else:
                    msn = 'minus_one_agressive'
                json_objects,reason_for_label,agree_count, dis_agree_count = building_data.avreage_direction_score_claculation(json_objects_change_minus1, TargetConcept, model, temperature, General_concepts, specific_concepts[d], concept_list, batch_size, debug=debug_current, iteration=iter_number, iteration_end=tot_iterations, test_flag=False, mission=msn,domain=d, agree_count=agree_count, dis_agree_count=dis_agree_count, preferance=preferance[0])
                #text = building_data.calculate_concept_importance(json_objects_change_minus1, TargetConcept,model_classification, temperature, General_concepts, specific_concepts[d], batch_size, debug=debug_current, iteration=iter_number, iteration_end=tot_iterations, test_flag=False, mission=msn, domain=d)
                #json_objects = json_utils.check_json_extraction(text, 2)
                #if json_objects == None:
                #    error_count += 1
                #    print("Error: JSON extraction failed. Please check the text.")
                #    continue
                #json_objects = json_utils.check_all_items_in_json(json_objects, 2,concept_list)
                #reason_for_label = {TargetConcept: json_objects[0][TargetConcept]} if TargetConcept in json_objects[0] else {}
                ##change Resonse1 of Response 1 to Chosen Response and Response2 to Rejected Response, since the labling model has no sense of preferance
                #for key, value in reason_for_label.items():
                #    value = value.replace('Response1', 'Chosen Response')
                #    value = value.replace('Response 1', 'Chosen Response')
                #    value = value.replace('Response2', 'Rejected Response')
                #    value = value.replace('Response 2', 'Rejected Response')
                #    reason_for_label[key] = value

                #check the data extracted:
                for index, item in json_objects.items():
                    #change all labbels -1 to 3:
                    for key, value in item.items():
                        if int(value) == -1:
                            item[key] = 3
                    #check the classifiction of the target concept, but only the one that was changed, if it matches the target label
                    index_in_batch = index
                    #find the currect item in the batch:
                    for key, b in json_objects_change_minus1[0].items():
                        if index_in_batch == str(b['Original Index']):
                            fit_batch = b
                            fit_batch['Concepts'] = item
                            break
                    
                    #I need to raise an error in the case midification was needed but the flag is false
                    for b in batch:
                        if index_in_batch == str(b['original_index']):
                            fit_batch2 = b
                            break
                    #find the original label of the concept
                    if TargetConcept in fit_batch2['General Concepts llm as judge'].keys():
                        original_label = int(fit_batch2['General Concepts llm as judge'][TargetConcept])
                    else:
                        original_label = int(fit_batch2['Specific Concepts llm as judge'][TargetConcept])
                    
                    new_label = int(item[TargetConcept])
                    current_label_for_chat.append(new_label)
                    #Case 1: the llm did not change the responses even when it was needed
                    if (not fit_batch["ChangeFlag"]) & (original_label != 3): #No change were made but the original label is not the target label
                        if new_label == original_label:
                            no_change_were_made_even_when_needed += 1
                            batch_for_recursion.append(fit_batch)
                            continue
                        #else: #Changes were made but the llm failed to report true on thr change flag
        
                    #Case 2" the llm changed the responses even when it was not needed
                    if (fit_batch["ChangeFlag"]) & (original_label == 3): #change were made but the original label is the target label
                        llm_unnecessry_modification += 1
                        continue
                    #Case 3: the llm changed the responses and it was needed
                    if ((fit_batch["ChangeFlag"]) or (new_label != original_label)) & (original_label != 3) : #change were made and the original label is not the target label
                        
                        #Case 3.1" the new label remains the same as the original label
                        if new_label == original_label:
                            batch_for_recursion.append(fit_batch)
                        #case 3.2: the lable changed from original in the opposite direction (from 0 to 1):
                        #in this case I dont want to save the change and call the modification agin with the original in batch 
                        elif (new_label == 1) and (original_label == 0):
                            batch_for_recursion.append(fit_batch2)
                        #case 3.3: the new label changed to somthing else then the original label
                        else: #I wanto to save what appens even if the label is not the target label
                            #updte the correct normalization factor:
                            General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
                            General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
                            General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                            General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                            General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1 = change_label_utils.update_normalizatopn_factors(new_label, original_label,General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2,
                                                                                                                                                                                                        Food_concept_norm_0_2, Legal_concept_norm_0_2,
                                                                                                                                                                                                        General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1,
                                                                                                                                                                                                        General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0,
                                                                                                                                                                                                        General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2,
                                                                                                                                                                                                        General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1,
                                                                                                                                                                                                        General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0,
                                                                                                                                                                                                        General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1)
                                                                                                                                                                                                    

                            #TO DO: call the modification agin
                            #TO DO: save the new dataset
                            for key, value in item.items():
                                if key == TargetConcept:
                                    continue
                                #find the original label of the concept
                                if key in fit_batch2['General Concepts llm as judge'].keys():
                                    original_label_current = int(fit_batch2['General Concepts llm as judge'][key])
                                else:
                                    original_label_current = int(fit_batch2['Specific Concepts llm as judge'][key])
                                current_label = int(value)
         
                                if current_label == original_label_current:
                                    continue
                                #extract the right dict to save the changes
                                relevent_dict = change_label_utils.dict_change_extraction(General_concepts, d,original_label, new_label, key,General_concept_dict_0_minus1,General_concept_dict_0_1,General_concept_dict_0_2,
                                                                       General_concept_dict_1_minus1,General_concept_dict_1_0,General_concept_dict_1_2,
                                                                       General_concept_dict_2_minus1,General_concept_dict_2_0,General_concept_dict_2_1,
                                                                       Food_concept_dict_0_minus1,Food_concept_dict_0_1,Food_concept_dict_0_2,
                                                                       Food_concept_dict_1_minus1,Food_concept_dict_1_0,Food_concept_dict_1_2,
                                                                       Food_concept_dict_2_minus1,Food_concept_dict_2_0,Food_concept_dict_2_1,
                                                                       Legal_concept_dict_0_minus1,Legal_concept_dict_0_1,Legal_concept_dict_0_2,
                                                                       Legal_concept_dict_1_minus1,Legal_concept_dict_1_0,Legal_concept_dict_1_2,
                                                                       Legal_concept_dict_2_minus1,Legal_concept_dict_2_0,Legal_concept_dict_2_1)
                                
                                if (original_label_current == 0) and (current_label == 1):
                                    relevent_dict[key]['0-1'] += 1
                                elif (original_label_current == 0) and (current_label == 2):
                                    relevent_dict[key]['0-2'] += 1
                                elif (original_label_current == 0) and (current_label == 3):
                                    relevent_dict[key]['0-3'] += 1
                                elif (original_label_current == 1) and (current_label == 0):
                                    relevent_dict[key]['1-0'] += 1
                                elif (original_label_current == 1) and (current_label == 2):
                                    relevent_dict[key]['1-2'] += 1
                                elif (original_label_current == 1) and (current_label == 3):
                                    relevent_dict[key]['1-3'] += 1
                                elif (original_label_current == 2) and (current_label == 0):
                                    relevent_dict[key]['2-0'] += 1
                                elif (original_label_current == 2) and (current_label == 1):
                                    relevent_dict[key]['2-1'] += 1
                                elif (original_label_current == 2) and (current_label == 3):
                                    relevent_dict[key]['2-3'] += 1
                                elif (original_label_current == 3) and (current_label == 0):
                                    relevent_dict[key]['3-0'] += 1
                                elif (original_label_current == 3) and (current_label == 1):
                                    relevent_dict[key]['3-1'] += 1
                                elif (original_label_current == 3) and (current_label == 2):
                                    relevent_dict[key]['3-2'] += 1
                                else:
                                    print(f"Error: The original label {original_label} and the current label {current_label} are not in the expected range.")
                                    error_count += 1
                                    continue
                            #save the new dataset: I need to add to fit_batch the labels for each concpets
                            data_to_save = fit_batch.copy()
                            new_data_to_save[d].append(data_to_save)
                            #case 3.2.1: the new label is not the target label
                            if new_label != 3:
                                batch_for_recursion.append(fit_batch)
                                #I will call the modification agin 
                                pass
                            #case 3.2.2: the new label is the target label, nothing else to do, I alresfy saved the dataset
                if len(batch_for_recursion) > 0:            
                    General_concept_dict_0_1, General_concept_dict_0_2, General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,\
                General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,\
                Food_concept_dict_0_1, Food_concept_dict_0_2, Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,\
                Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,\
                Legal_concept_dict_0_1, Legal_concept_dict_0_2, Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,\
                Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
                General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
                General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                    General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                        General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,agree_count, dis_agree_count = building_data.recursive_change_minus_one_specific_items_history_chat(batch_for_recursion,concept_list, General_concepts, specific_concepts, d, TargetConcept, General_concept_dict_0_minus1,General_concept_dict_0_1,General_concept_dict_0_2,
                                                                        General_concept_dict_1_minus1,General_concept_dict_1_0,General_concept_dict_1_2,
                                                                        General_concept_dict_2_minus1,General_concept_dict_2_0,General_concept_dict_2_1,
                                                                        Food_concept_dict_0_minus1,Food_concept_dict_0_1,Food_concept_dict_0_2,
                                                                        Food_concept_dict_1_minus1,Food_concept_dict_1_0,Food_concept_dict_1_2,
                                                                        Food_concept_dict_2_minus1,Food_concept_dict_2_0,Food_concept_dict_2_1,
                                                                        Legal_concept_dict_0_minus1,Legal_concept_dict_0_1,Legal_concept_dict_0_2,
                                                                        Legal_concept_dict_1_minus1,Legal_concept_dict_1_0,Legal_concept_dict_1_2,
                                                                        Legal_concept_dict_2_minus1,Legal_concept_dict_2_0,Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, 
                                                                            General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, 
                                                                            General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, 
                                                                            General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,chat_session,current_label_for_chat, reason_for_label, model_classification, temperature, batch_size, debug=debug_current, iter_number=iter_number, itreration_end = tot_iterations, trails = 2, agressive=agressive_change,agree_count = agree_count, dis_agree_count = dis_agree_count)            
                                
                
                                      
                iter_number += 1
                batch_for_recursion = []
            except Exception as e:
                traceback.print_exc()
                error_count += 1
                batch_for_recursion = []
                print(f"Error: An exception occurred during processing. Error count: {error_count}")
                if error_count > 100:
                    print("Error: Too many errors occurred. Exiting.")
                    break
        change_label_utils.save_change_dict(output_path,General_concept_dict_0_minus1,General_concept_dict_0_1,General_concept_dict_0_2,
                                                                          General_concept_dict_1_minus1,General_concept_dict_1_0,General_concept_dict_1_2,
                                                                          General_concept_dict_2_minus1,General_concept_dict_2_0,General_concept_dict_2_1,
                                                                          Food_concept_dict_0_minus1,Food_concept_dict_0_1,Food_concept_dict_0_2,
                                                                          Food_concept_dict_1_minus1,Food_concept_dict_1_0,Food_concept_dict_1_2,
                                                                          Food_concept_dict_2_minus1,Food_concept_dict_2_0,Food_concept_dict_2_1,
                                                                          Legal_concept_dict_0_minus1,Legal_concept_dict_0_1,Legal_concept_dict_0_2,
                                                                          Legal_concept_dict_1_minus1,Legal_concept_dict_1_0,Legal_concept_dict_1_2,
                                                                          Legal_concept_dict_2_minus1,Legal_concept_dict_2_0,Legal_concept_dict_2_1, d, iter_number, trails = 1, agrresive = agressive_change,key = TargetConcept, domain = d, model = model, batch_size = batch_size)
        #save the new dataset
        if not agressive_change:
            if len(new_data_to_save[d]) > 0:
                with open(os.path.join(new_data_path, f'Model_{MODEL_NAME}_new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{1}_number_of_exampels_{batch_size}.json'), 'w') as f:
                    json.dump(new_data_to_save[d], f, indent=4)
        else:
            if len(new_data_to_save[d]) > 0:
                with open(os.path.join(new_data_path, f'Model_{MODEL_NAME}new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{1}__number_of_exampels_{batch_size}_aggresive.json'), 'w') as f:
                    json.dump(new_data_to_save[d], f, indent=4)
    #save the mismatch tracking for each domain
        with open(os.path.join(output_path, f'mismatch_tracking_{d}.json'), 'w') as f:
            json.dump(mismatch_tracking[d], f, indent=4)
        print(f"Total items processed in domain {d}: {item_count}")
        print(f"Total mismatches in domain {d}: {mismatch}")
        print(f"Total errors in llm as judge label in domain {d}: {error_in_llm_as_judge_label}")
    
    print(f"Total items processed: {item_count}")
    print(f"Total mismatches: {mismatch}")
    print(f"Total errors: {error_count}")
            
   
    print("Phase One: change the labels of the concepts so all will be 1 (Represebt thw chosen concept more then the rejected one)")                
              
               
if __name__ == "__main__":
    main()
