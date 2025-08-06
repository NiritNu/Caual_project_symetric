import math
import os
import sys
import traceback
import random
import json
#from vertexai import generative_models
from vertexai.preview.generative_models import GenerativeModel

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
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
output_path = os.path.join(BASE_DIR,'code', 'out_symmetric_data')
new_data_path = os.path.join(BASE_DIR,'code', 'out_symmetric_data', 'new_data_to_save')
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
    model_classification = GenerativeModel(model_name=MODEL_NAME)
    temperature = 0.5
    # Load the data
    trainset, testset, General_concepts, specific_concepts = preprocessing_data(ChangeminusOne = True)
    # create concept list of all the concepts
    concept_list = list(General_concepts.keys()) + list(specific_concepts['food'].keys()) + list(specific_concepts['legal'].keys())
        
    # Phase One: change the labels of the concepts so all will be 1 (Represebt thw chosen concept more then the rejected one)
    #domains = ['food', 'legal']
    domains = ['legal']  # Change this to 'legal' if needed
    index_for_debug_food = []#+ [i for i in range(100, 110)]
    index_for_debug_legal = []#[i for i in range(60)]
    batch_size = 1
   
  
    for d in domains:  
        data_to_save = []  
        lenght_of_trainset = math.ceil(len(trainset[d]))
        item_count = 0
        error_count = 0
        iter_number = 0
        tot_iterations = lenght_of_trainset // batch_size
        agree_count = 0
        dis_agree_count = 0
        while iter_number < tot_iterations:
            try:
                # restart conversation for each iteration
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
               
                
                # preferamce calculation:
                preferance = building_data.llmAsJudgeAvreageForBaseData(batch, model_classification, temperature, batch_size, debug=debug_current, iteration=iter_number, iteration_end=tot_iterations, test_flag=False, domain=d, op = output_path)

                json_objects,agree_count, dis_agree_count = building_data.avreage_direction_score_claculationBaseData(batch, model_classification, temperature, General_concepts,specific_concepts[d], concept_list, batch_size, debug=debug_current, iteration=iter_number, iteration_end=tot_iterations, test_flag=False,domain=d, agree_count=agree_count, dis_agree_count=dis_agree_count, preferance=preferance[0], op=output_path)
                #update data to save
                base_data = batch[0]
                base_data['preference'] = preferance[0]
                #split json_objects to teo dirctionries ,General and Specific concepts
                Genenral_concepts_dict = {}
                Specific_concepts_dict = {}
                for item in json_objects.items():
                    for key, value in item[1].items():
                        if key in General_concepts:
                            Genenral_concepts_dict[key] = value
                        elif key in specific_concepts[d]:
                            Specific_concepts_dict[key] = value
                base_data['General Concepts llm as judge'] = Genenral_concepts_dict
                base_data['Specific Concepts llm as judge'] = Specific_concepts_dict
                data_to_save.append(base_data)
                
                iter_number += 1

                
            except Exception as e:
                traceback.print_exc()
                error_count += 1
                print(f"Error: An exception occurred during processing. Error count: {error_count}")
                if error_count > 100:
                    print("Error: Too many errors occurred. Exiting.")
                    break
        #save the data
        if len(data_to_save) > 0:
            file_name = f'Model{MODEL_NAME_BASE}_Domain_{d}_FullInfo_symetric_postprocessing_train.txt'
            file_path = os.path.join(output_path, file_name)
            with open(file_path, "w") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
    
    print("Phase One: change the labels of the concepts so all will be 1 (Represebt thw chosen concept more then the rejected one)")                
              
               
if __name__ == "__main__":
    main()
