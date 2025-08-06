import math
import os
import sys
import random
import json
import copy
#from langchain_google_vertexai import VertexAI
from vertexai import generative_models

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from utils import data_utils, prompt_utils, json_utils
import verification

# Constants
BASE_DIR = '/home/nirit/IBMProject'
FOOD_PATH = os.path.join(BASE_DIR, 'food_organized_indices.json')
FOOD_CONCEPT_PATH = os.path.join(BASE_DIR, 'food_organized_concepts.json')
LEGAL_PATH = os.path.join(BASE_DIR, 'legal_organized_indices.json')     
LEGAL_CONCEPT_PATH = os.path.join(BASE_DIR, 'legal_organized_concepts.json')
PARENT_PROMPT_PATH = os.path.join(BASE_DIR, 'code_symetric', 'prompts')
BASE_PROMPT_PATH = [os.path.join(PARENT_PROMPT_PATH, 'LLMasAjudge.txt'), os.path.join(PARENT_PROMPT_PATH, 'ExtractGeneralConcepts.txt')
                    , os.path.join(PARENT_PROMPT_PATH, 'ExtractAddtionalGeneralConcepts.txt'), os.path.join(PARENT_PROMPT_PATH, 'GeneralConceptsClassificationNoPreferanceNoTargeConcept.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ExtractSpecificConcepts.txt'), os.path.join(PARENT_PROMPT_PATH, 'ExtractAddtionalSpecificConcepts.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'FilterGeneralConcepts.txt'), os.path.join(PARENT_PROMPT_PATH, 'ExtractGeneralConceptsEqual.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ExtractAdditionalGeneralEqualConcepts.txt')]

output_path = os.path.join(BASE_DIR,'code_symetric', 'output_concepts')
if not os.path.exists(output_path):
    os.makedirs(output_path)

TOPIC_LIST = [FOOD_PATH, LEGAL_PATH]
CONCEPT_LIST = [FOOD_CONCEPT_PATH, LEGAL_CONCEPT_PATH]
NAMES_LIST = ['food', 'legal']

# google models
Home = False
if not Home:
    path_to_creds = r"/home/nirit/IBMProject/HowToRunForMNitay/gemma-nlp-422508-c4d4ce542552-gilat.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds
else:
    key = data_utils.load_text("/home/nirit/gemini/run.txt")
    #if ends with \n, remove it
    if key.endswith('\n'):
        key = key[:-1]
    os.environ["GOOGLE_API_KEY"] = key

MODEL_NAME = "gemini-2.0-flash-001" #"gemini-pro" #chevk which is better

def set_seed(seed):
    random.seed(seed)

def llmAsJudge(data_dict, model, temperature, n=2, sampeld_index = [], debug=False, iteration=0, test_flag=False, slice=None, domain_number = 2, per_example=False, statistics=None):

    ## Important Note: Chosen and Rejected by the original data from Nitay, not from my process .
    name_to_save = f'Model{MODEL_NAME}_llmAsJudge_number_of_examples{n}_iteration_{iteration}.json'
    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_Test_llmAsJudge_number_of_examples{n}_iteration_{iteration}.json'

    if not debug:
        #sample data
        if test_flag:
            sampled_examples, slice = data_utils.sample_data_by_order(data_dict, domain_number*n, slice)
            sampeld_keys = []
        else:
            sampled_examples , sampeld_keys = data_utils.sample_data(data_dict, n, sampeld_index)
            # save sampled keys for later use
            with open(os.path.join(output_path, f'sampled_index_{name_to_save}'), 'w') as f:
                json.dump(sampeld_keys, f)
        json_objects = {
            'food': [],
            'legal': []
        } #This is hard coded for the food and legal domains, change it if you want to use other domains.
        error_count = 0
        for item in sampled_examples:
            
            prompt_chosen = prompt_utils.createPromptLLMasJudge(BASE_PROMPT_PATH[0],[item], direction='Chosen first')
            prompt_rejected = prompt_utils.createPromptLLMasJudge(BASE_PROMPT_PATH[0],[item], direction='Rejected first')

            #feed to model stage llm as a judge - #TO DO: add major vote to preferance, as a loop
            try:
                response_chosen = model.generate_content(prompt_chosen, generation_config={"temperature": temperature})
                response_rejected = model.generate_content(prompt_rejected, generation_config={"temperature": temperature})
            except Exception as e:
                print(f"Error occurred during model generation: {e}")
                error_count += 1
                if error_count > 5:
                    print("Too many errors, stopping the process.")
                    return None, None, None, statistics
                
                continue
            text_chosen = response_chosen.candidates[0].content.parts[0].text
            text_rejected = response_rejected.candidates[0].content.parts[0].text
            json_objects_chosen = check_json_extraction(text_chosen, 1)
            json_objects_rejected = check_json_extraction(text_rejected, 1)
            # average the json objects:
            json_objects_tmp = json_utils.average_json_objects(json_objects_chosen[0], json_objects_rejected[0], case='llmAsJudge')
            if json_objects_tmp is None:
                print("Error: JSON extraction failed. Please check the text.")
                continue
            for k,v in json_objects_tmp[0].items(): #ToDO this is for one example change if it will be for more
                json_objects[k].append(v[0])
            
    #save json objects:
        with open(os.path.join(output_path, name_to_save), 'w') as f:
            json.dump(json_objects, f, indent=2)
            #save the response
   
        
    else:
        with open(os.path.join(output_path, name_to_save), 'r') as f:
                json_objects = json.load(f)
        if slice is not None:
            slice = slice + domain_number*n
            sampeld_keys = []
        else:
            with open(os.path.join(output_path, f'sampled_index_{name_to_save}'), 'r') as f:
                sampeld_keys = json.load(f)
        #debug:

    for key, value in json_objects.items():
        for value2 in value:
            if 'preference' not in value2:
                print(f"Error: preference not found in {key} {value2}")
                continue
            if value2['preference'] == 1:
                statistics['y=1'] += 1
            elif value2['preference'] == 3:
                statistics['y=3'] += 1
            elif value2['preference'] == 0:
                statistics['y=0'] += 1
            else:
                print(f"Error: preference {value2['preference']} not recognized in {key} {value2}")
    print(f"Statistics for iteration {iteration}: {statistics}")
    print(f"sampled keys len: {len(sampeld_keys)}")
    return json_objects, slice, sampeld_keys

def General_concepts_extraction(json_objects, model,General_concepts_number , temperature, n,debug=False, iteration=0):
    if not debug:
        prompt_general_concepts1, prompt_general_concepts2 , prompt_general_equal1,prompt_general_equal2 = prompt_utils.createPromptExtractGeneral(BASE_PROMPT_PATH[1], BASE_PROMPT_PATH[7], json_objects,General_concepts_number)

        response1 = model.generate_content(prompt_general_concepts1, generation_config={"temperature": temperature})
        text1 = response1.candidates[0].content.parts[0].text
        response2 = model.generate_content(prompt_general_concepts2, generation_config={"temperature": temperature})
        text2 = response2.candidates[0].content.parts[0].text
        response_equal1 = model.generate_content(prompt_general_equal1, generation_config={"temperature": temperature})
        text_equal1 = response_equal1.candidates[0].content.parts[0].text
        response_equal2 = model.generate_content(prompt_general_equal2, generation_config={"temperature": temperature})
        text_equal2 = response_equal2.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text1, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConcepts_number_of_examples{n}_iteration_{iteration}_1.txt'))
        data_utils.save_text(text2, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConcepts_number_of_examples{n}_iteration_{iteration}_2.txt'))
        data_utils.save_text(text_equal1, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsEqual_number_of_examples{n}_iteration_{iteration}_1.txt'))
        data_utils.save_text(text_equal2, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsEqual_number_of_examples{n}_iteration_{iteration}_2.txt'))
    else:
        text1 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConcepts_number_of_examples{n}_iteration_{iteration}_1.txt'))
        text2 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConcepts_number_of_examples{n}_iteration_{iteration}_2.txt'))
        text_equal1 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsEqual_number_of_examples{n}_iteration_{iteration}_1.txt'))
        text_equal2 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsEqual_number_of_examples{n}_iteration_{iteration}_2.txt'))
    return text1, text2, text_equal1, text_equal2

def Generl_concepts_addition(json_objects, model,General_concepts_number,prev_general_concepts,prev_general_concepts_equal , temperature, n,debug=False, iteration=0):
            
    def process_data_iteration(json_objects, general_concept_number,general_concepts_prev,prev_general_concepts_equal, iteration=0):
        """Processes data and pre-existing concepts with the LLM."""
        # Create the prompt for the current iteration
        content, content2, contentEqual, contentEqual2 = prompt_utils.createPromptExtractGeneral(BASE_PROMPT_PATH[2],BASE_PROMPT_PATH[8], json_objects, general_concept_number)
        content += "Pre-Existing Concepts:\n"
        content2 += "Pre-Existing Concepts:\n"
        content += json.dumps(general_concepts_prev, indent=2)  # Add the previous concepts in JSON format
        content2 += json.dumps(general_concepts_prev, indent=2)  # Add the previous concepts in JSON format
        contentEqual += "Pre-Existing Concepts:\n"
        contentEqual2 += "Pre-Existing Concepts:\n"
        contentEqual += json.dumps(prev_general_concepts_equal, indent=2)  # Add the previous concepts in JSON format
        contentEqual2 += json.dumps(prev_general_concepts_equal, indent=2)  # Add the previous concepts in JSON format

        # Add domain-specific concepts
        response = model.generate_content(content, generation_config={"temperature": temperature})
        response2 = model.generate_content(content2, generation_config={"temperature": temperature})
        response_equal = model.generate_content(contentEqual, generation_config={"temperature": temperature})
        response2_equal = model.generate_content(contentEqual2, generation_config={"temperature": temperature})

        return response, response2, response_equal, response2_equal

    if not debug:
    
        #extract  addtional general concepts:
        response, response2, response_equal, response2_equal = process_data_iteration(json_objects, General_concepts_number, prev_general_concepts, prev_general_concepts_equal, iteration)
        text = response.candidates[0].content.parts[0].text
        text2 = response2.candidates[0].content.parts[0].text
        text_equal = response_equal.candidates[0].content.parts[0].text
        text2_equal = response2_equal.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_1.txt'))
        data_utils.save_text(text2, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_2.txt'))
        data_utils.save_text(text_equal, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_equal_1.txt'))
        data_utils.save_text(text2_equal, os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_equal_2.txt'))
    else:
        text = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_1.txt'))
        text2 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_2.txt'))
        text_equal = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_equal_1.txt'))
        text2_equal = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_GeneralConceptsAddition_iteration_{iteration}_number_of_examples{n}_equal_2.txt'))

    return text, text2, text_equal, text2_equal

def Specific_concepts_extraction(json_objects, model,concepts_number,general_concepts , temperature, n,debug=False, iteration=0, domain_name='No_Domain'):
            
    def process_data_iteration(json_objects, concept_number,general_concepts, iteration=0):
        """Processes data and pre-existing concepts with the LLM."""
        # Create the prompt for the current iteration
        content1,content2 = prompt_utils.createPromptExtractSpecific(BASE_PROMPT_PATH[4], json_objects, concept_number)
        content1 += "Pre-Existing General Concepts:\n"
        content2 += "Pre-Existing General Concepts:\n"
        content1 += json.dumps(general_concepts, indent=2)  # Add the previous concepts in JSON format
        content2 += json.dumps(general_concepts, indent=2)  # Add the previous concepts in JSON format

        # Add domain-specific concepts
        response1 = model.generate_content(content1, generation_config={"temperature": temperature})
        response2 = model.generate_content(content2, generation_config={"temperature": temperature})
        return response1, response2

    if not debug:
    
        #extract  addtional general concepts:
        response1, response2 = process_data_iteration(json_objects, concepts_number, general_concepts, iteration)
        text1 = response1.candidates[0].content.parts[0].text
        text2 = response2.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text1, os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{domain_name}_SpecificConcepts_iteration_{iteration}_number_of_examples{n}_1.txt'))
        data_utils.save_text(text2, os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{domain_name}_SpecificConcepts_iteration_{iteration}_number_of_examples{n}_2.txt'))
    else:
        text1 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{domain_name}_SpecificConcepts_iteration_{iteration}_number_of_examples{n}_1.txt'))
        text2 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{domain_name}_SpecificConcepts_iteration_{iteration}_number_of_examples{n}_2.txt'))

    return text1, text2

def Specific_concepts_addition(json_objects, model,concepts_number,general_concepts, prev_specific_concepts , temperature, n,debug=False, iteration=0, domain_name='No_Domain'):
            
    def process_data_iteration(json_objects, concept_number,general_concepts,prev_specific_concepts, iteration=0):
        """Processes data and pre-existing concepts with the LLM."""
        # Create the prompt for the current iteration
        content1, content2 = prompt_utils.createPromptExtractSpecific(BASE_PROMPT_PATH[5], json_objects, concept_number)
        content1 += "Pre-Existing Concepts:\n"
        content2 += "Pre-Existing Concepts:\n"
        content1 += json.dumps(general_concepts, indent=2)  #
        content2 += json.dumps(general_concepts, indent=2)  #
        content1 += "Pre-Existing Specific Concepts:\n"
        content2 += "Pre-Existing Specific Concepts:\n"
        content1 += json.dumps(prev_specific_concepts, indent=2)  # Add the previous concepts in JSON format
        content2 += json.dumps(prev_specific_concepts, indent=2)  # Add the previous concepts in JSON format

        # Add domain-specific concepts
        response1 = model.generate_content(content1, generation_config={"temperature": temperature})
        response2 = model.generate_content(content2, generation_config={"temperature": temperature})
        return response1, response2

    if not debug:
    
        #extract  addtional general concepts:
        response1, respnse2 = process_data_iteration(json_objects, concepts_number, general_concepts,prev_specific_concepts, iteration)
        text1 = response1.candidates[0].content.parts[0].text
        text2 = respnse2.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text1, os.path.join(output_path,f'Model{MODEL_NAME}__Domain_{domain_name}_SpecificConceptsAddition_iteration_{iteration}_number_of_examples{n}_1.txt'))
        data_utils.save_text(text2, os.path.join(output_path,f'Model{MODEL_NAME}__Domain_{domain_name}_SpecificConceptsAddition_iteration_{iteration}_number_of_examples{n}_2.txt'))
    else:
        text1 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}__Domain_{domain_name}_SpecificConceptsAddition_iteration_{iteration}_number_of_examples{n}_1.txt'))
        text2 = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}__Domain_{domain_name}_SpecificConceptsAddition_iteration_{iteration}_number_of_examples{n}_2.txt'))

    return text1, text2

def calculate_concept_importance(json_objects,model, temperature ,concepts_list,n,output_path, debug,iteration,iteration_end = None ,test_flag=False, food_index_to_concepts_llm_as_judge = [], legal_index_to_concepts_llm_as_judge = [], equal_concepts = False, General = True):

    json_object = {'dataset': {}}
    error_count = 0
    equal = 'equal_concepts' if equal_concepts else 'not_equal_concepts'
    for d,value in json_objects.items():
        for i, item in enumerate(value):
            should_retry = True
            while should_retry:
                should_retry = False
                if General:
                    if test_flag:
                        name_to_save_chosen = f'Model{MODEL_NAME}_dommain_{d}_GeneralConceptsClassification_Test_chosen_iteration_{iteration}_item_{i}_number_of_examples{n}_{equal}.txt'
                        name_to_save_rejected = f'Model{MODEL_NAME}_domain_{d}_GeneralConceptsClassification_Test_rejected_iteration_{iteration}_item_{i}_number_of_examples{n}_{equal}.txt'
                    else:
                        name_to_save_chosen = f'Model{MODEL_NAME}_domain_{d}_GeneralConceptsClassification_chosen_iteration_{iteration}_out_of_{iteration_end}_item_{i}_number_of_examples{n}_{equal}.txt'
                        name_to_save_rejected = f'Model{MODEL_NAME}_domain_{d}_GeneralConceptsClassification_rejected_iteration_{iteration}_out_of_{iteration_end}_item_{i}_number_of_examples{n}_{equal}.txt'
                else:#Specific concepts
                    if test_flag:
                        name_to_save_chosen = f'Model{MODEL_NAME}_domain_{d}_SpecificConceptsClassification_Test_chosen_iteration_{iteration}_item_{i}_number_of_examples{n}_{equal}.txt'
                        name_to_save_rejected = f'Model{MODEL_NAME}_domain_{d}_SpecificConceptsClassification_Test_rejected_iteration_{iteration}_item_{i}_number_of_examples{n}_{equal}.txt'
                    else:
                        name_to_save_chosen = f'Model{MODEL_NAME}_domain_{d}_SpecificConceptsClassification_chosen_iteration_{iteration}_out_of_{iteration_end}_item_{i}_number_of_examples{n}_{equal}.txt'
                        name_to_save_rejected = f'Model{MODEL_NAME}_domain_{d}_SpecificConceptsClassification_rejected_iteration_{iteration}_out_of_{iteration_end}_item_{i}_number_of_examples{n}_{equal}.txt'
                if not debug:
                    #prompt for the model
                    prompt_chosen, prompt_rejected = prompt_utils.createPromptGeneralConceptClassification(BASE_PROMPT_PATH[3], item, concepts_list, General = General)
                    try:
                        response_chosen = model.generate_content(prompt_chosen, generation_config={"temperature": temperature})
                        response_rejected = model.generate_content(prompt_rejected, generation_config={"temperature": temperature})
                    except Exception as e:
                        print(f"Error occurred during concepts classification check: {e}")
                        error_count += 1
                        if error_count > 5:
                            print("Too many errors, stopping the process.")
                            return None, None, None
                        continue
                    text_chosen = response_chosen.candidates[0].content.parts[0].text
                    text_rejected = response_rejected.candidates[0].content.parts[0].text
                    #save the response
                    data_utils.save_text(text_chosen, os.path.join(output_path,name_to_save_chosen))
                    data_utils.save_text(text_rejected, os.path.join(output_path,name_to_save_rejected))
                else:
                    text_chosen = data_utils.load_text(os.path.join(output_path,name_to_save_chosen))
                    text_rejected = data_utils.load_text(os.path.join(output_path,name_to_save_rejected))

                try: #for debug, delete this
                    json_objects_chosen = check_json_extraction(text_chosen, 1)
                    json_objects_rejected = check_json_extraction(text_rejected, 1)
                except Exception as e:
                    print(f"Error occurred during JSON extraction: {e}")
                    error_count += 1
                    if error_count > 5:
                        print("Too many errors, stopping the process.")
                        return None, None, None
                    should_retry = True
                    continue
                if (json_utils.check_all_items_in_json(json_objects_chosen, 3) == None)or (json_utils.check_all_items_in_json(json_objects_rejected, 3) == None):
                    print("Error: JSON extraction failed. Please check the text.")
                    error_count += 1
                    if error_count > 5:
                        print("Too many errors, stopping the process.")
                        return None, None, None
                    should_retry = True
                    continue
                json_objects_tmp = json_utils.average_json_objects(json_objects_chosen[0], json_objects_rejected[0], case='concept_classification')
                json_object['dataset'] = {**json_object['dataset'], **json_objects_tmp[0]['dataset']}
                if json_objects == None:
                    print("Error: JSON extraction failed. Please check the text.")
                    continue
                #update the general concepts dict: 
                json_items = list(json_objects_tmp[0]['dataset'].items())
                if d == 'food':
                    food_index_to_concepts_llm_as_judge += json_items
                elif d == 'legal':
                    legal_index_to_concepts_llm_as_judge +=json_items
    json_object = [json_object]

    return json_object, food_index_to_concepts_llm_as_judge, legal_index_to_concepts_llm_as_judge

def calculate_specific_concept_importance(json_objects,model, temperature ,concepts_list,n,output_path, debug,iteration, domain_name='No_Domain', iteration_end = None, test_flag=False, food_index_to_concepts_llm_as_judge = [], legal_index_to_concepts_llm_as_judge = []):
    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_SpecificConceptsClassification_Test_Domain_{domain_name}_iteration_{iteration}_number_of_examples{n}.txt'
    else:
        name_to_save = f'Model{MODEL_NAME}_SpecificConceptsClassification_Domain_{domain_name}_iteration_{iteration}_out_of_{iteration_end}_number_of_examples{n}.txt'
    if not debug:
        Procced = True
        error_count = 0
        while Procced:
            #prompt for the model
            prompt1, prompt2 = prompt_utils.createPromptSpecificConceptClassification(BASE_PROMPT_PATH[3], json_objects[0], concepts_list)
            response1 = model.generate_content(prompt1, generation_config={"temperature": temperature})
            response2 = model.generate_content(prompt2, generation_config={"temperature": temperature})
            text1 = response1.candidates[0].content.parts[0].text
            text2 = response2.candidates[0].content.parts[0].text
            #check the extraction
            json_objects_chosen = check_json_extraction(text1, 1)
            json_objects_rejected = check_json_extraction(text2, 1)
            if (json_objects_chosen == None) or (json_objects_rejected == None):
                print("Error: JSON extraction failed. Please check the text.")
                error_count += 1
                if error_count > 5:
                    print("Too many errors, stopping the process.")
                    return None, None
                continue
            Procced = False
            json_objects = json_utils.average_json_objects(json_objects_chosen[0], json_objects_rejected[0], case='concept_classification')
            #save the json objects
            with open(os.path.join(output_path, name_to_save), 'w') as f:
                json.dump(json_objects, f, indent=2)
    else:
        with open(os.path.join(output_path, name_to_save), 'r') as f:
            json_objects = json.load(f)

    #update the general concepts dict: 
    json_items = list(json_objects[0]['dataset'].items())
    if d == 'food':
        food_index_to_concepts_llm_as_judge += json_items
    elif d == 'legal':
        legal_index_to_concepts_llm_as_judge +=json_items
                

    return [json_objects]

def filter_concepts(concepts_dict, model,temperature, debug = False, iteration=0):
    if not debug:
        #filter the concepts based on the threshold
        prompt = prompt_utils.createPromptFilterConcepts(BASE_PROMPT_PATH[6], concepts_dict)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path,f'Model{MODEL_NAME}_FilterConcepts2_iteration_{iteration}.txt'))
    else:
        text = data_utils.load_text(os.path.join(output_path,f'Model{MODEL_NAME}_FilterConcepts2_iteration_{iteration}.txt'))
    return text

def filter_concept_by_thresh(prev_general_concepts,model,n,temperature, debug = False, iteration_start = 0, iteration_end = 0, equal_concepts = False):
    equal = 'equal_concepts' if equal_concepts else 'not_equal_concepts'
    if not debug:
        # building the count dictionary for the specific concepts:
        basic_dict = {k: 0 for k in ['0','1', '2', '3']} #labels for the specific concepts
        concepts_dict = {k: copy.deepcopy(basic_dict) for k in prev_general_concepts.keys()} #for each specific concept
        preference_dict = {k:copy.deepcopy(concepts_dict) for k in ['0','1','3']} #for each specific concept
        number_of_examples_0 = 0 #For debug
        number_of_examples_1 = 0 #For debug
        number_of_examples_3 = 0 #For debug

        general_concepts_dict = {k: 0 for k in prev_general_concepts.keys()}
        general_concepts_dict_equal = {k: 0 for k in prev_general_concepts.keys()}
        general_concepts_dict_not_equal = {k: 0 for k in prev_general_concepts.keys()}
        index_to_upload_debug = [] #For debug
        number_of_examples = 0
        number_of_examples_equal = 0 #For debug
        number_of_examples_not_equal = 0 #For debug
        minus_one_count = 0 #For debug
        minus_one_count_equal = 0 #For debug
        minus_one_count_not_equal = 0 #For debug
        two_count = 0 #For debug
        two_count_equal = 0 #For debug
        two_count_not_equal = 0 #For debug
        zero_count = 0 #For debug
        zero_count_equal = 0 #For debug
        zero_count_not_equal = 0 #For debug
        one_count = 0 #For debug
        one_count_equal = 0 #For debug
        one_count_not_equal = 0 #For debug
        error_count = 0
        #Hard coded for food and legal:
        food_index_to_concepts_llm_as_judge  = []
        legal_index_to_concepts_llm_as_judge  = []
        for j in range(iteration_start, iteration_end+1):
            print(f" General Concepts Classification :: Iteration {j+1} of {iteration_end+1}")
            # load the llm as judge dataset
            if j in index_to_upload_debug:
                debug_current = True
            else:
                debug_current = False
            #read json file:
            json_objects_examples = data_utils.load_json(os.path.join(output_path, f'Model{MODEL_NAME}_llmAsJudge_number_of_examples{n}_iteration_{j}.json'))
            #balance the labels:
            json_objects_examples,_ = data_utils.balamce_data_per_domain(json_objects_examples)

            json_objects , food_index , legal_index = calculate_concept_importance(json_objects_examples, model, temperature ,prev_general_concepts,n ,output_path, debug_current, j, test_flag=False, iteration_end=iteration_end, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge, equal_concepts=equal_concepts)
            if json_objects == None:
                print("Error: JSON extraction failed. Please check the text.")
                error_count += 1
                if error_count > 5:
                    print("Too many errors, stopping the process filter concept by threshold.")
                    return None
                continue
            #food_index_to_concepts_llm_as_judge  += food_index
            #legal_index_to_concepts_llm_as_judge  += legal_index

            '''for key, value in json_objects[0]['dataset'].items():
                number_of_examples += 1
                competable_item = data_utils.extract_item_by_original_index(json_objects_examples, key)
                preference = competable_item['preference'] 
                if preference == 0:
                    number_of_examples_0 += 1
                    dictionary_to_update = preference_dict['0']
                    add_equal = True
                elif preference == 1:
                    number_of_examples_1 += 1
                    dictionary_to_update = preference_dict['1']
                    add_equal = False
                elif preference == 3:
                    number_of_examples_3 += 1
                    dictionary_to_update = preference_dict['3']
                    add_equal = False
                else:
                    print(f"Error: Invalid preference value {preference} for key {key}.")
                    continue
                for key2, value2 in value.items():
                    if key2 in dictionary_to_update:
                        if value2 == 1:
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            dictionary_to_update[key2]['1'] += 1
                            one_count += 1
                        elif value2 == 3: #For debug
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            dictionary_to_update[key2]['3'] += 1
                            minus_one_count += 1
                        elif value2 == 2: #For debug
                            dictionary_to_update[key2]['2'] += 1
                            two_count += 1  
                        elif value2 == 0: #For debug
                            dictionary_to_update[key2]['0'] += 1
                            if add_equal:
                                general_concepts_dict_equal[key2] += 1
                            zero_count += 1''' #MAybe change to this in the future

            for key, value in json_objects[0]['dataset'].items():
                number_of_examples += 1
                competable_item = data_utils.extract_item_by_original_index(json_objects_examples, key)
                preference = competable_item['preference'] 
                if preference == 0:
                    number_of_examples_equal += 1
                    add_equal = True
                else:
                    number_of_examples_not_equal += 1
                    add_equal = False
                for key2, value2 in value.items():
                    if key2 in general_concepts_dict:
                        if value2 == 1:
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            one_count += 1
                        elif value2 == 3: #For debug
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            minus_one_count += 1
                        elif value2 == 2: #For debug
                            two_count += 1  
                        elif value2 == 0: #For debug
                            if add_equal:
                                general_concepts_dict_equal[key2] += 1
                            zero_count += 1

            print(f"minus one count: {minus_one_count}") #For debug
            print(f"two count: {two_count}") #For debug
            print(f"zero count: {zero_count}") #For debug
            print(f"one count: {one_count}") #For debug
        
        
            #normalize the counts:
        '''for key, value in preference_dict.items():
            # if key is '0' normlaize all values in the iiner dictonries by number_of_examples_0
            # if key is '1' normlaize all values in the iiner dictonries by number_of_examples_1
            # if key is '3' normlaize all values in the iiner dictonries by number_of_examples_3
            if key == '0':
                for key2, value2 in value.items():
                    if number_of_examples_0 > 0:
                        value2['0'] = value2['0'] / number_of_examples_0
                        value2['1'] = value2['1'] / number_of_examples_0
                        value2['2'] = value2['2'] / number_of_examples_0
                        value2['3'] = value2['3'] / number_of_examples_0
            elif key == '1':
                for key2, value2 in value.items():
                    if number_of_examples_1 > 0:
                        value2['0'] = value2['0'] / number_of_examples_1
                        value2['1'] = value2['1'] / number_of_examples_1
                        value2['2'] = value2['2'] / number_of_examples_1
                        value2['3'] = value2['3'] / number_of_examples_1
            elif key == '3':
                for key2, value2 in value.items():
                    if number_of_examples_3 > 0:
                        value2['0'] = value2['0'] / number_of_examples_3
                        value2['1'] = value2['1'] / number_of_examples_3
                        value2['2'] = value2['2'] / number_of_examples_3
                        value2['3'] = value2['3'] / number_of_examples_3
        
        total_score_per_concept = {k: 0 for k in prev_general_concepts.keys()}
        total_score_per_concept_not_zero = {k: 0 for k in prev_general_concepts.keys()}
        
        for key, value in preference_dict.items():
            if key == '0':
                number_of_examples_key = number_of_examples_0
            elif key == '1':
                number_of_examples_key = number_of_examples_1
            elif key == '3':
                number_of_examples_key = number_of_examples_3
            for key2, value2 in value.items():
                total_score_per_concept[key2] += value2[key]/number_of_examples*number_of_examples_key
                if key != '0':
                    total_score_per_concept_not_zero[key2] += value2[key]/(number_of_examples_1+number_of_examples_3)*number_of_examples_key
''' #MAy be change to this in the future
        #calculate precentage of each concept
        for key, value in general_concepts_dict.items():
            general_concepts_dict[key] = value / number_of_examples
        for key, value in general_concepts_dict_equal.items():
            general_concepts_dict_equal[key] = value / number_of_examples_equal
        for key, value in general_concepts_dict_not_equal.items():
            general_concepts_dict_not_equal[key] = value / number_of_examples_not_equal

    
        # keep only the concepts that are above threshold
        threshold_equal = 0.7
        threshold_not_equal = 0.35
        general_concepts_dict_equal = {k: v for k, v in general_concepts_dict_equal.items() if v > threshold_equal}
        general_concepts_dict_not_equal = {k: v for k, v in general_concepts_dict_not_equal.items() if v > threshold_not_equal}
        #keep  a dictonry with the concepts that are in both dictionaries
        general_concepts_dict_to_return = {k: prev_general_concepts[k] for k, _ in general_concepts_dict.items() if k in general_concepts_dict_equal and k in general_concepts_dict_not_equal}
        #save the dict:
        with open(os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_number_of_exampels_{n}_itertion{iteration_end}_{equal}.json'), 'w') as f:
            json.dump(general_concepts_dict_to_return, f, indent=2)
    else:
        #upload saved dictionary
        general_concepts_dict_to_return = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_number_of_exampels_{n}_itertion{iteration_end}_{equal}.json'))
    return general_concepts_dict_to_return

def classify_concepts_on_test_set(prev_general_concepts,model,n,temperature,test_data_dict):
    #Fof General concepts classification:
    
    general_concepts_dict = {k: 0 for k in prev_general_concepts.keys()}
    general_concepts_dict_equal = {k: 0 for k in prev_general_concepts.keys()}
    general_concepts_dict_not_equal = {k: 0 for k in prev_general_concepts.keys()}
    index_to_upload_debug = [] #For debug
    number_of_examples = 0
    number_of_examples_not_equal = 0 #For debug
    number_of_examples_equal = 0 #For debug
    minus_one_count = 0 #For debug
    two_count = 0 #For debug
    zero_count = 0 #For debug
    one_count = 0 #For debug
    food_index_to_concepts_llm_as_judge = []
    legal_index_to_concepts_llm_as_judge = []
    #Hard coded for food and legal:
    number_of_exnamples = 0
    number_of_domains = len(test_data_dict)
    for key, value in test_data_dict.items():
        number_of_exnamples += len(value)
    iterations = math.ceil(number_of_exnamples/(number_of_domains*n))
    i = 0
    slice = 0
    while i < iterations:
        print(f" General Concepts Classification Test Time:: Iteration {i+1} of {iterations}")
        # load the llm as judge dataset
        if i in index_to_upload_debug:
            debug_current = True
        else:
            debug_current = False
        test_statistics = {'y=1': 0, 'y=3': 0, 'y=0': 0} #For debug
        json_objects_examples, slice, _ = llmAsJudge(test_data_dict, model, temperature, n=n, debug=debug_current, iteration=i+1, test_flag=True, slice = slice, domain_number = number_of_domains, statistics=test_statistics)
        if json_objects_examples == None:
            print("Error: JSON extraction failed. Please check the text.")
            slice = slice - number_of_domains*n
            continue
        json_objects_examples,_ = data_utils.balamce_data_per_domain(json_objects_examples)

        json_objects , food_index , legal_index = calculate_concept_importance(json_objects_examples, model, temperature ,prev_general_concepts,n ,output_path, debug_current, i, test_flag=True, iteration_end=None, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge, equal_concepts=False)
        if json_objects == None:
            print("Error: JSON extraction failed. Please check the text.")
            error_count += 1
            if error_count > 5:
                print("Too many errors, stopping the process filter concept by threshold.")
                return None
            continue

        for key, value in json_objects[0]['dataset'].items():
            number_of_examples += 1
            competable_item = data_utils.extract_item_by_original_index(json_objects_examples, key)
            preference = competable_item['preference'] 
            if preference == 0:
                number_of_examples_equal += 1
                add_equal = True
            else:
                number_of_examples_not_equal += 1
                add_equal = False
            for key2, value2 in value.items():
                if key2 in general_concepts_dict:
                    if value2 == 1:
                        general_concepts_dict[key2] += 1
                        if not add_equal:
                            general_concepts_dict_not_equal[key2] += 1
                        one_count += 1
                    elif value2 == 3: #For debug
                        general_concepts_dict[key2] += 1
                        if not add_equal:
                            general_concepts_dict_not_equal[key2] += 1
                        minus_one_count += 1
                    elif value2 == 2: #For debug
                        two_count += 1  
                    elif value2 == 0: #For debug
                        if add_equal:
                            general_concepts_dict_equal[key2] += 1
                        zero_count += 1

        print(f"minus one count: {minus_one_count}") #For debug
        print(f"two count: {two_count}") #For debug
        print(f"zero count: {zero_count}") #For debug
        print(f"one count: {one_count}") #For debug
        i += 1

    
    #calculate precentage of each concept
    for key, value in general_concepts_dict.items():
        general_concepts_dict[key] = value / number_of_examples
    for key, value in general_concepts_dict_equal.items():
        general_concepts_dict_equal[key] = value / number_of_examples_equal
    for key, value in general_concepts_dict_not_equal.items():
        general_concepts_dict_not_equal[key] = value / number_of_examples_not_equal

    # keep only the concepts that are above threshold
    threshold_equal = 0.7
    threshold_not_equal = 0.35
    general_concepts_dict_equal = {k: v for k, v in general_concepts_dict_equal.items() if v > threshold_equal}
    general_concepts_dict_not_equal = {k: v for k, v in general_concepts_dict_not_equal.items() if v > threshold_not_equal}
    prev_general_concepts = {k: v for k, v in prev_general_concepts.items() if (k in general_concepts_dict) }
    #keep  a dictonry with the concepts that are in both dictionaries
    general_concepts_dict_to_return = {k: prev_general_concepts[k] for k, _ in general_concepts_dict.items() if k in general_concepts_dict_equal and k in general_concepts_dict_not_equal}
    #save the dict:

    with open(os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_Test_number_of_exampels_{n}_itertion{i}.json'), 'w') as f:
        json.dump(general_concepts_dict_to_return, f, indent=2)

    #NOTE this part do in debug mode only, self filter the concepts that looks the same
    #Concepts to save:
    c2s = ['Efficiency','Helpfulness','Credibility','Specificity']
    prev_general_concepts = {k: v for k, v in prev_general_concepts.items() if k in c2s}

    return prev_general_concepts, food_index_to_concepts_llm_as_judge, legal_index_to_concepts_llm_as_judge

def filter_specific_concept_by_thresh(prev_specific_concepts,domain_sub_datast,model,example_number,n_specific, d,temperature, debug = False, iteration_start = 0, iteration_end = 0):

    # building the count dictionary for the specific concepts:
    basic_dict = {k: 0 for k in ['0','1', '2', '3']} #labels for the specific concepts
    concepts_dict = {k: copy.deepcopy(basic_dict) for k in prev_specific_concepts[d].keys()} #for each specific concept
    preference_dict = {k:copy.deepcopy(concepts_dict) for k in ['0','1','3']} #for each specific concept

    #filter specific concepts:
    #specific_concepts_dict = {k: 0 for k in prev_specific_concepts[d].keys()}
    #specific_concepts_dict_equal = {k: 0 for k in prev_specific_concepts[d].keys()}
    #specific_concepts_dict_not_equal = {k: 0 for k in prev_specific_concepts[d].keys()}

    #example_count_dict = {k: 1 for k in prev_specific_concepts[d].keys()} #start with 1 to avoid division by zero
    number_of_examples_0 = 0 #For debug
    number_of_examples_1 = 0 #For debug
    number_of_examples_3 = 0 #For debug
    index_to_upload_debug_domain_classification = {} 
    index_to_upload_debug_domain_classification['food'] = [i for i in range(31)] #For debug
    index_to_upload_debug_domain_classification['legal'] = [] #For debug
    number_of_examples = 0
    minus_one_count = 0 #For debug
    two_count = 0 #For debug
    zero_count = 0 #For debug
    one_count = 0 #For debug
    food_index_to_concepts_llm_as_judge_domain  = []
    legal_index_to_concepts_llm_as_judge_domain  = []
    for j in range(iteration_start, iteration_end+1):
        print(f" Specific Concepts Classification :: Domain {d} Iteration {j+1} of {iteration_end+1}")
        # load the llm as judge dataset
        if j in index_to_upload_debug_domain_classification[d]:
            debug_current = True
        else:
            debug_current = False
        #load the data:
        strart_index = j*n_specific
        if strart_index + n_specific > example_number[d]:
            end_index = example_number[d]
        else:
            end_index = strart_index + n_specific
        d_v = domain_sub_datast[d][strart_index:end_index]
        #It is already balanced with respect to the labels, so no need to balance it again 
        #classification_json = calculate_specific_concept_importance(d_v, model, temperature ,prev_specific_concepts[d],n_specific ,output_path, debug_current, j, d, iteration_end+1, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge_domain, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge_domain)
        json_objects , food_index , legal_index = calculate_concept_importance({d:d_v}, model, temperature ,prev_specific_concepts[d],n_specific ,output_path, debug_current, j, test_flag=False, iteration_end=iteration_end, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge_domain, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge_domain, equal_concepts=False, General = False)
        if json_objects == None:
            print("Error: JSON extraction failed. Please check the text.")
            error_count += 1
            if error_count > 5:
                print("Too many errors, stopping the process filter concept by threshold.")
                return None
            continue

        for key, value in json_objects[0]['dataset'].items():
            number_of_examples += 1
            competable_item = data_utils.extract_item_by_original_index({d:d_v}, key)
            preference = competable_item['preference'] 
            if preference == 0:
                number_of_examples_0 += 1
                dictionary_to_update = preference_dict['0']
            elif preference == 1:
                number_of_examples_1 += 1
                dictionary_to_update = preference_dict['1']
            elif preference == 3:
                number_of_examples_3 += 1
                dictionary_to_update = preference_dict['3']
            else:
                print(f"Error: Invalid preference value {preference} for key {key}.")
                continue
            for key2, value2 in value.items():
                if key2 in dictionary_to_update:
                    if value2 == 1:
                        dictionary_to_update[key2]['1'] += 1
                        one_count += 1
                    elif value2 == 3: #For debug
                        dictionary_to_update[key2]['3'] += 1
                        minus_one_count += 1
                    elif value2 == 2: #For debug
                        dictionary_to_update[key2]['2'] += 1
                        two_count += 1  
                    elif value2 == 0: #For debug
                        dictionary_to_update[key2]['0'] += 1
                        zero_count += 1

        print(f"minus one count: {minus_one_count}") #For debug
        print(f"two count: {two_count}") #For debug
        print(f"zero count: {zero_count}") #For debug
        print(f"one count: {one_count}") #For debug
    #normalize the counts:
    for key, value in preference_dict.items():
        # if key is '0' normlaize all values in the iiner dictonries by number_of_examples_0
        # if key is '1' normlaize all values in the iiner dictonries by number_of_examples_1
        # if key is '3' normlaize all values in the iiner dictonries by number_of_examples_3
        if key == '0':
            for key2, value2 in value.items():
                if number_of_examples_0 > 0:
                    value2['0'] = value2['0'] / number_of_examples_0
                    value2['1'] = value2['1'] / number_of_examples_0
                    value2['2'] = value2['2'] / number_of_examples_0
                    value2['3'] = value2['3'] / number_of_examples_0
        elif key == '1':
            for key2, value2 in value.items():
                if number_of_examples_1 > 0:
                    value2['0'] = value2['0'] / number_of_examples_1
                    value2['1'] = value2['1'] / number_of_examples_1
                    value2['2'] = value2['2'] / number_of_examples_1
                    value2['3'] = value2['3'] / number_of_examples_1
        elif key == '3':
            for key2, value2 in value.items():
                if number_of_examples_3 > 0:
                    value2['0'] = value2['0'] / number_of_examples_3
                    value2['1'] = value2['1'] / number_of_examples_3
                    value2['2'] = value2['2'] / number_of_examples_3
                    value2['3'] = value2['3'] / number_of_examples_3

    total_score_per_concept = {k: 0 for k in prev_specific_concepts[d].keys()}
    total_score_per_concept_not_zero = {k: 0 for k in prev_specific_concepts[d].keys()}

    for key, value in preference_dict.items():
        if key == '0':
            number_of_examples_key = number_of_examples_0
        elif key == '1':
            number_of_examples_key = number_of_examples_1
        elif key == '3':
            number_of_examples_key = number_of_examples_3
        for key2, value2 in value.items():
            total_score_per_concept[key2] += value2[key]/number_of_examples*number_of_examples_key
            if key != '0':
                total_score_per_concept_not_zero[key2] += value2[key]/(number_of_examples_1+number_of_examples_3)*number_of_examples_key
        '''for key, value in classification_json[0].items():
            number_of_examples += 1
            for key2, value2 in value.items():
                if key2 in specific_concepts_dict:
                    if value2 == '1':
                        specific_concepts_dict[key2] += 1
                        example_count_dict[key2] += 1#count all exmaples that are not 2
                        one_count += 1
                    elif value2 == '-1': #For debug
                        example_count_dict[key2] += 1#count all exmaples that are not 2
                        minus_one_count += 1
                    elif value2 == '2': #For debug
                        two_count += 1
                    elif value2 == '0': #For debug
                        example_count_dict[key2] += 1 #count all exmaples that are not 2
                        zero_count += 1
        print(f"two count: {two_count}") #For debug
        print(f"zero count: {zero_count}") #For debug
        print(f"one count: {one_count}") #For debug
        print(f"minus one count: {minus_one_count}")'''

    
    #calculate precentage of each concept
    '''for key, value in specific_concepts_dict.items():
        specific_concepts_dict[key] = value / example_count_dict[key]

    ratio_of_appreance = {key: value / number_of_examples for key, value in example_count_dict.items()}'''

    # keep only the concepts that are above threshold
    threshold = 0.32
    total_score_per_concept = {k: v for k, v in total_score_per_concept.items() if v > threshold}
    #keep onlu concepts that appear in the ratio_of_appreance more than 0.5:
    #ratio_of_appreance = {k: v for k, v in ratio_of_appreance.items() if v > 0.5}

    prev_specific_concepts[d] = {k: v for k, v in prev_specific_concepts[d].items() if (k in total_score_per_concept) }

    return prev_specific_concepts


def classify_specific_concepts_on_test_set(prev_specific_concepts,model,n,temperature,test_data_dict,d):
    
    
    #filter specific concepts:
    specific_concepts_dict = {k: 0 for k in prev_specific_concepts.keys()}
    example_count_dict = {k: 1 for k in prev_specific_concepts.keys()} #start with 1 to avoid division by zero
    index_to_upload_debug_domain_classification = {} 
    food_index_to_concepts_llm_as_judge_domain = []
    legal_index_to_concepts_llm_as_judge_domain = []
    if d == 'food':
        index_to_upload_debug_domain_classification = [i for i in range(30)]
    elif d == 'legal':
        index_to_upload_debug_domain_classification = []
            
    number_of_examples = 0
    minus_one_count = 0 #For debug
    two_count = 0 #For debug
    zero_count = 0 #For debug
    one_count = 0 #For debug
    #Hard coded for food and legal:
    number_of_exnamples = 0

    #if test_data_dict is a dict of dicts, convert it to a list of dict
    if isinstance(test_data_dict, dict):
        test_data_dict = list(test_data_dict.values())
    number_of_exnamples += len(test_data_dict)
    iterations = math.ceil(number_of_exnamples/(n))
    if d == 'food':
        iterations_saved = [i for i in range(1,11)]
    elif d == 'legal':
        iterations_saved = [i for i in range(11,21)]
    i = 0
    while i < iterations:
        print(f" Specific Concepts Classification :: Domain {d} Iteration {i+1} of {iterations}")
        # load the llm as judge dataset
        if i in index_to_upload_debug_domain_classification:
            debug_current = True
        else:
            debug_current = False
        #load the data:
        #load llm as judge data
        file_name = f'Model{MODEL_NAME}_llmAsJudge_Test_number_of_examples{5}_iteration_{iterations_saved[i]}.txt'
        text = data_utils.load_text(os.path.join(output_path,file_name))
        json_objects = check_json_extraction(text, 1)
        if json_objects == None:
            print("Error: JSON extraction failed. Please check the text.")
            continue
        text = calculate_specific_concept_importance(json_objects[0][d], model, temperature ,prev_specific_concepts,n ,output_path, debug_current, i, d, test_flag=True)
        classification_json = check_json_extraction(text, 1)
        if classification_json == None:
            print("Error: JSON extraction failed. Please check the text.")
            continue

        json_items = list(classification_json[0].items())
        if d == 'food':
                food_index_to_concepts_llm_as_judge_domain += json_items
        elif d == 'legal':
                legal_index_to_concepts_llm_as_judge_domain += json_items
           
        for key, value in classification_json[0].items():
            number_of_examples += 1
            for key2, value2 in value.items():
                if key2 in specific_concepts_dict:
                    if value2 == '1':
                        specific_concepts_dict[key2] += 1
                        example_count_dict[key2] += 1#count all exmaples that are not 2
                        one_count += 1
                    elif value2 == '-1': #For debug
                        example_count_dict[key2] += 1#count all exmaples that are not 2
                        minus_one_count += 1
                    elif value2 == '2': #For debug
                        two_count += 1
                    elif value2 == '0': #For debug
                        example_count_dict[key2] += 1 #count all exmaples that are not 2
                        zero_count += 1
        print(f"two count: {two_count}") #For debug
        print(f"zero count: {zero_count}") #For debug
        print(f"one count: {one_count}") #For debug
        print(f"minus one count: {minus_one_count}")
        i += 1

    
    #calculate precentage of each concept
    for key, value in specific_concepts_dict.items():
        specific_concepts_dict[key] = value / example_count_dict[key]

    ratio_of_appreance = {key: value / number_of_examples for key, value in example_count_dict.items()}

    #save the dict:
    with open(os.path.join(output_path,f'Model{MODEL_NAME}_specific_concepts_Domain_{d}_dict_Test_number_of_exampels_{n}_itertion{i}.json'), 'w') as f:
        json.dump(specific_concepts_dict, f, indent=2)
    #save ratio_of_appreance:
    with open(os.path.join(output_path,f'Model{MODEL_NAME}_ratio_of_appreance_Domain_{d}_Test_number_of_exampels_{n}_itertion{i}.json'), 'w') as f:
        json.dump(ratio_of_appreance, f, indent=2)
    # keep only the concepts that are above threshold
    threshold = 0.75
    
    specific_concepts_dict = {k: v for k, v in specific_concepts_dict.items() if v > threshold}
    #keep onlu concepts that appear in the ratio_of_appreance more than 0.5:
    ratio_of_appreance = {k: v for k, v in ratio_of_appreance.items() if v > 0.5}

    prev_specific_concepts = {k: v for k, v in prev_specific_concepts.items() if (k in specific_concepts_dict) and (k in ratio_of_appreance)}

    with open(os.path.join(output_path,f'Model{MODEL_NAME}_specific_concepts_dict_Domain_{d}_Test_number_of_exampels_{n}_itertion{i}.json'), 'w') as f:
        json.dump(prev_specific_concepts, f, indent=2)

    return prev_specific_concepts, food_index_to_concepts_llm_as_judge_domain, legal_index_to_concepts_llm_as_judge_domain









def check_json_extraction(text, number_of_expected_json):
    json_objects,_ = json_utils.json_extract(text)

    #check if the json objects are valid, if not manuallu exytact the json objects
    if not verification.number_of_jsons_extracted(json_objects,number_of_expected_json): # not two json files...somethong went wrong and I want to extract the text using "json", and END JSON
        text = json_utils.fix_json_files_if_extrcted_failed(text)
        if text == None:
            print("Error: JSON extraction failed. Please check the text.")
            return None
        elif text == 'json\n' + '\nEND JSON\n': #No json were extracted and there is nothiing to fix, need a new iteration
            print("Error: No JSON objects were found in the text.")
            return None
        else:    
            json_objects = json_utils.json_extract(text)
    
    return json_objects

def main():
    #init seed
    set_seed(24)

    # init model
    model = generative_models.GenerativeModel(model_name=MODEL_NAME)
    temperature = 0.5

    # Load the data
    domains = [TOPIC_LIST, NAMES_LIST]
    domain_number = len(domains[1])
    data_dict = data_utils.creating_data_dict(domains)
    #split to train and test
    data_dict, test_data_dict = data_utils.split_data(data_dict, 0.75)
    #debug : a = data_utils.remove_duplicates_debug_concepts(data_dict,'Original_index')
    #       a = data_utils.remove_duplicates_debug_concepts(test_data_dict,'Original_index')
    #       data_utils.train_test_overlap_check(data_dict, test_data_dict)
    n = 5
    statistics = {'y=1': 0,'y=3': 0, 'y=0': 0} #for debug
    statistics_balanced = statistics.copy() #for debug
    json_objects,_, sampled_index = llmAsJudge(data_dict, model, temperature, n=n, debug=True, iteration=0, per_example = True, statistics=statistics)
    json_objects, statistics_balanced_batch = data_utils.balamce_data_per_domain(json_objects)
    statistics_balanced = {key: statistics_balanced[key] + statistics_balanced_batch[key] for key in statistics_balanced}
    #extract the json file:
    # Find all JSON objects within the text.
    ## Important Note: Chosen and Rejected by the original data from Nitay, not from my process .

    #extract general concepts:
    #prompt for general concepts:
    General_concepts_number = '4'
    first_iteration = True
    error_count_first = 0
    while first_iteration:
        text1, text2, text_equal1, text_equal2 = General_concepts_extraction(json_objects, model,General_concepts_number , temperature, n=n, debug=True, iteration=0)
        json_objects1 = check_json_extraction(text1, 1)
        json_objects2 = check_json_extraction(text2, 1)
        json_objects3 = check_json_extraction(text_equal1, 1)
        json_objects4 = check_json_extraction(text_equal2, 1)
        if (json_objects1 == None) or (json_objects2 == None) or (json_objects3 == None) or (json_objects4 == None):
            print("Error: JSON extraction failed. Please check the text.")
            error_count_first += 1
            if error_count_first > 5:
                print("Too many errors, stopping the process.")
                return None, None, None
            continue
        first_iteration = False
    prev_general_concepts = {**json_objects1[0], **json_objects2[0]}
    prev_general_concepts_equal = {**json_objects3[0], **json_objects4[0]}    
    # loop over the general concepts

     # create promt to stage 2 : initratvly change the concepts based on previous concepts list and their ranking
    iter_num = 59
    filter_concepts_frequency_unite = 20000
    filter_concepts_frequency_th = 10 # to do if needed

    i = 0
    index_to_upload_debug = [i for i in range(11)] #For debug
    count_to_break = 0
    General = False
    if General:
        while i < iter_num:
            try: 
                print(f" General Concepts :: Iteration {i+1} of {iter_num}")
                if i in index_to_upload_debug:
                    debug_current = True
                else:
                    debug_current = False
                # llm as judge for new examples
                previous_sampled_index = sampled_index.copy()
                
                json_objects, _, sampled_index = llmAsJudge(data_dict, model, temperature, n=n, sampeld_index=sampled_index, debug=debug_current, iteration=i+1, statistics=statistics_balanced.copy(), per_example=True)
                #balance the data per domain
                json_objects, statistics_balanced_tmp = data_utils.balamce_data_per_domain(json_objects)
                statistics_balanced = {key: statistics_balanced[key] + statistics_balanced_tmp[key] for key in statistics_balanced}

                if json_objects == None:
                    print(f"Error occurred during llmAsJudge")
                    sampled_index = previous_sampled_index
                    continue
                
                
                text1, text2, text3, text4 = Generl_concepts_addition(json_objects, model,General_concepts_number, prev_general_concepts, prev_general_concepts_equal , temperature, n=n, debug=debug_current, iteration=i+1)
                    
                #verify the response
                json_objects1 = check_json_extraction(text1, 1)
                json_objects2 = check_json_extraction(text2, 1)
                json_objects3 = check_json_extraction(text3, 1)
                json_objects4 = check_json_extraction(text4, 1)
                if (json_objects1 == None) or (json_objects2 == None)or (json_objects3 == None) or (json_objects4 == None):
                    print("Error: JSON extraction failed. Please check the text.")
                    sampled_index = previous_sampled_index
                    continue

                json_objects = [{**json_objects1[0], **json_objects2[0]}]  # merge the two json objects
                json_objects_equal = [{**json_objects3[0], **json_objects4[0]}]  # merge the two json objects
              
                # adding new concept to previous. while making sure no prevous concept is removed, if the description changes, change is
                current_general_concepts = json_objects[0]
                for key, value in current_general_concepts.items():
                    if key in prev_general_concepts:
                        if prev_general_concepts[key] != value:
                            prev_general_concepts[key] = value
                    else:
                        prev_general_concepts[key] = value
                current_general_concepts_equal = json_objects_equal[0]
                for key, value in current_general_concepts_equal.items():
                    if key in prev_general_concepts_equal:
                        if prev_general_concepts_equal[key] != value:
                            prev_general_concepts_equal[key] = value
                    else:
                        prev_general_concepts_equal[key] = value

                
                if (i % filter_concepts_frequency_th ==0) and (i !=0):

                    prev_general_concepts = filter_concept_by_thresh(prev_general_concepts,model,n,temperature,debug = debug_current,iteration_start=0, iteration_end=i, equal_concepts=False)
                    if prev_general_concepts == None:
                        print("Error: JSON extraction failed. Please check the text.")
                        error_count += 1
                        if error_count > 5:
                            print("Too many errors, stopping the process.")
                            return None, None, None

                        continue
                    '''prev_general_concepts_equal = filter_concept_by_thresh(prev_general_concepts_equal,model,n,temperature,debug = debug_current,iteration_start=0, iteration_end=i, equal_concepts=True)
                    if prev_general_concepts_equal == None:
                        print("Error: JSON extraction failed. Please check the text.")
                        error_count += 1
                        if error_count > 5:
                            print("Too many errors, stopping the process.")
                            return None, None, None

                        continue'''

                    

                #if not verification.run_all_verification(json_objects,2,int(general_concepts)*(i+2),int(specific_concepts)*domain_number*(i+2) ,specific_concepts):
                #    print("Error: Verification failed. Please check the JSON objects.")
                #    continue
                #else:
                #    print("Verification passed.")
                i += 1
                # calculate the importance of the concepts and extract the concepts of prevous run (if the verification process failed, I will not get the concepts of this run)
                #general_concepts_prev, domain_concepts_prev = calculate_concept_importance(json_objects, output_path, i+1)

                # Filtering the concepts: for each pair of responses, I want to see if the concept represent betterr the chosen
                
                # upload all llm as judge dataset I save earlier
            #convert prev_general_concepts to a dictionary, with the keys as the name of the concepta and zero as value to all keys
            except Exception as e:
                count_to_break +=1
                sampled_index = previous_sampled_index #if the extraction failed, I want to keep the previous sampled index
                print("Error: Something went wrong in General extracition. tring agin.")
                if count_to_break > 30:
                    print("Error: Too many errors. Exiting.")#
                    break
                else:
                    continue

        
            
        #filter same concepts:
        
        
        general_concepts_dict = {k: 0 for k in prev_general_concepts.keys()}
        general_concepts_dict_equal = {k: 0 for k in prev_general_concepts.keys()}
        general_concepts_dict_not_equal = {k: 0 for k in prev_general_concepts.keys()}
        index_to_upload_debug = [i for i in range(61)] #For debug
        number_of_examples = 0
        number_of_examples_not_equal = 0
        number_of_examples_equal = 0
        minus_one_count = 0 #For debug
        two_count = 0 #For debug
        zero_count = 0 #For debug
        one_count = 0 #For debug
        #Hard coded for food and legal:
        food_index_to_concepts_llm_as_judge  = []
        legal_index_to_concepts_llm_as_judge  = []
        for j in range(0, i+1):
            print(f" General Concepts Classification :: Iteration {j+1} of {i+1}")
            # load the llm as judge dataset
            if j in index_to_upload_debug:
                debug_current = True
            else:
                debug_current = False
            #read json file:
            json_objects_examples = data_utils.load_json(os.path.join(output_path, f'Model{MODEL_NAME}_llmAsJudge_number_of_examples{n}_iteration_{j}.json'))
            #balance the labels:
            json_objects_examples,_ = data_utils.balamce_data_per_domain(json_objects_examples)

            json_objects , food_index , legal_index = calculate_concept_importance(json_objects_examples, model, temperature ,prev_general_concepts,n ,output_path, debug_current, j, test_flag=False, iteration_end=i+1, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge, equal_concepts=False)

            if json_objects == None:
                print("Error: JSON extraction failed. Please check the text.")
                error_count += 1
                if error_count > 5:
                    print("Too many errors, stopping the process filter concept by threshold.")
                    return None
                continue
            #food_index_to_concepts_llm_as_judge  += food_index
            #legal_index_to_concepts_llm_as_judge  += legal_index

            for key, value in json_objects[0]['dataset'].items():
                number_of_examples += 1
                competable_item = data_utils.extract_item_by_original_index(json_objects_examples, key)
                preference = competable_item['preference'] 
                if preference == 0:
                    number_of_examples_equal += 1
                    add_equal = True
                else:
                    number_of_examples_not_equal += 1
                    add_equal = False
                for key2, value2 in value.items():
                    if key2 in general_concepts_dict:
                        if value2 == 1:
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            one_count += 1
                        elif value2 == 3: #For debug
                            general_concepts_dict[key2] += 1
                            if not add_equal:
                                general_concepts_dict_not_equal[key2] += 1
                            minus_one_count += 1
                        elif value2 == 2: #For debug
                            two_count += 1  
                        elif value2 == 0: #For debug
                            if add_equal:
                                general_concepts_dict_equal[key2] += 1
                            zero_count += 1

            print(f"minus one count: {minus_one_count}") #For debug
            print(f"two count: {two_count}") #For debug
            print(f"zero count: {zero_count}") #For debug
            print(f"one count: {one_count}") #For debug

        #calculate precentage of each concept
        for key, value in general_concepts_dict.items():
            general_concepts_dict[key] = value / number_of_examples
        for key, value in general_concepts_dict_equal.items():
            general_concepts_dict_equal[key] = value / number_of_examples_equal
        for key, value in general_concepts_dict_not_equal.items():
            general_concepts_dict_not_equal[key] = value / number_of_examples_not_equal
            
        #keep only the concepts that are above threshold
        threshold_equal = 0.7
        threshold_not_equal = 0.35
        general_concepts_dict_equal = {k: v for k, v in general_concepts_dict_equal.items() if v > threshold_equal}
        general_concepts_dict_not_equal = {k: v for k, v in general_concepts_dict_not_equal.items() if v > threshold_not_equal}
        #keep  a dictonry with the concepts that are in both dictionaries
        general_concepts_dict_to_return = {k: prev_general_concepts[k] for k, _ in general_concepts_dict.items() if k in general_concepts_dict_equal and k in general_concepts_dict_not_equal}
        #save the dict:
        equal = 'not_equal'
        with open(os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_number_of_exampels_{n}_itertion{i+1}_{equal}.json'), 'w') as f:
            json.dump(general_concepts_dict_to_return, f, indent=2)
   

        #Test the general concepts on the test set
        #change test_data_dict to be a dict of lists:
        data_dict_test_lists = {}
        for key, value in test_data_dict.items():
            data_dict_test_lists[key] = []
            for value2 in value.values():
                data_dict_test_lists[key].append(value2)

        data_utils.save_text(json.dumps(prev_general_concepts, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_prev_number_of_exampels_{n}_before_Test_filter.json'))

        
        prev_general_concepts, food_test_generl_concepts, legal_test_general_concepts = classify_concepts_on_test_set(prev_general_concepts,model,n,temperature,data_dict_test_lists)
        

        #save the general concepts dict
        data_utils.save_text(json.dumps(general_concepts_dict, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_number_of_exampels_{n}.json'))
        #save the general concepts dict, only the ones that are above the threshold
        data_utils.save_text(json.dumps(prev_general_concepts, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_prev_number_of_exampels_{n}.json'))

        #sabe llm as judge for food and legal general concepts
        food_index_to_concepts_llm_as_judge = dict(food_index_to_concepts_llm_as_judge)
        legal_index_to_concepts_llm_as_judge = dict(legal_index_to_concepts_llm_as_judge)
        food_index_to_concepts_llm_test = dict(food_test_generl_concepts)
        legal_index_to_concepts_llm_test = dict(legal_test_general_concepts)
        data_utils.save_text(json.dumps(food_index_to_concepts_llm_as_judge, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_food_index_to_concepts_llm_as_judge.json'))
        data_utils.save_text(json.dumps(legal_index_to_concepts_llm_as_judge, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_legal_index_to_concepts_llm_as_judge.json'))
        data_utils.save_text(json.dumps(food_index_to_concepts_llm_test, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_food_index_to_concepts_llm_as_judge_Test.json'))
        data_utils.save_text(json.dumps(legal_index_to_concepts_llm_test, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_legal_index_to_concepts_llm_as_judge_Test.json'))

    else:
        food_index_to_concepts_llm_as_judge = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_food_index_to_concepts_llm_as_judge.json')) 
        legal_index_to_concepts_llm_as_judge = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_legal_index_to_concepts_llm_as_judge.json'))
        food_index_to_concepts_llm_test = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_food_index_to_concepts_llm_as_judge_Test.json'))
        legal_index_to_concepts_llm_test = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_legal_index_to_concepts_llm_as_judge_Test.json'))

        prev_general_concepts = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_general_concepts_dict_prev_number_of_exampels_{n}.json'))

    ## adding domain specific concepts
    # add the first domain specific
    ############################# END OF GENERAL CONCEPTS ##########################################################
    ############################# START OF SPECIFIC CONCEPTS ##########################################################
    
    #upload all datasets per domain
    domain_sub_datast = {'train':{}, 'test':{}}
    example_number = {}
        #add to test set the general concepst

    for i in range(20):
        json_objects = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_Test_llmAsJudge_number_of_examples{n}_iteration_{i+1}.json'))
        for domain_name, value in json_objects.items():
            #add to value th llm as judge per General concepts
            for value2 in value:
                original_index = value2['original_index']
                if domain_name == 'food':
                    value2['General Concepts llm as judge'] = food_index_to_concepts_llm_test[str(original_index)]
                elif domain_name == 'legal':
                    value2['General Concepts llm as judge'] = legal_index_to_concepts_llm_test[str(original_index)]
            if domain_name not in domain_sub_datast['test']:
                domain_sub_datast['test'][domain_name] = []
            for value2 in value:
                domain_sub_datast['test'][domain_name].append(value2)
    #train set
    for i in range(iter_num):
        json_objects = data_utils.load_json(os.path.join(output_path,f'Model{MODEL_NAME}_llmAsJudge_number_of_examples{n}_iteration_{i}.json'))
        for domain_name, value in json_objects.items():
            #add to value th llm as judge per General concepts
            for value2 in value:
                original_index = value2['original_index']
                if domain_name == 'food':
                    value2['General Concepts llm as judge'] = food_index_to_concepts_llm_as_judge[str(original_index)]
                elif domain_name == 'legal':
                    value2['General Concepts llm as judge'] = legal_index_to_concepts_llm_as_judge[str(original_index)]
            if domain_name not in domain_sub_datast['train']:
                domain_sub_datast['train'][domain_name] = []
                example_number[domain_name] = 0
            example_number[domain_name] += len(value)
            for value2 in value:
                domain_sub_datast['train'][domain_name].append(value2)     
    
    Domain_concepts_number = '3'
    n_specific = n*2
    # extract initial domain concepts
    json_objects_specific = {}
    prev_specific_concepts = {}
    index_to_upload_debug = {}
    index_to_upload_debug['food'] = [i for i in range(31)] #For debug
    index_to_upload_debug['legal'] = [] #For debug

    filter_specific_concepts_frequency_th = 10

    
    for d in domain_sub_datast['train'].keys(): #go through each domain seperately
        iter_num_specific = math.ceil(example_number[d]/n_specific)
        error_count = 0
        for si in range(iter_num_specific):
            keep_goning_specific = True
            while keep_goning_specific:
                print(f" Specific Concepts Extraction :: Domain {d} Iteration {si+1} of {iter_num_specific}")
                # llm as judge for new examples
                if si in index_to_upload_debug[d]:
                    debug_current = True
                else:
                    debug_current = False
                strart_index = si*n_specific
                if strart_index + n_specific > example_number[d]:
                    end_index = example_number[d]
                else:
                    end_index = strart_index + n_specific
                d_v = domain_sub_datast['train'][d][strart_index:end_index] 
                #balance the data set :

                d_v, statistics = data_utils.balamce_data_per_domain({d: d_v})
                if si == 0: #first iteration
                    text1,text2 = Specific_concepts_extraction(d_v, model,Domain_concepts_number,prev_general_concepts , temperature, n_specific,debug=debug_current, iteration=si, domain_name=d)
                else:
                    text1,text2 = Specific_concepts_addition(d_v, model,Domain_concepts_number,prev_general_concepts, prev_specific_concepts[d] , temperature, n_specific,debug=debug_current, iteration=si, domain_name=d)
                json_objects1 = check_json_extraction(text1, 1)
                json_objects2 = check_json_extraction(text2, 1)
                if (json_objects1 == None) or (json_objects2 == None):
                    print("Error: JSON extraction failed. Please check the text.")
                    error_count += 1
                    if error_count > 5:
                        print("Too many errors, stopping the process.")
                        return None, None, None
                    continue
                keep_goning_specific = False
                json_objects_specific[d] = {**json_objects1[0], **json_objects2[0]}  # merge the two json objects
                #updating concepts, without deleting the previous concepts, and updating the descrption if needed
                current_specific_concepts = json_objects_specific[d]
                if d not in prev_specific_concepts:
                    prev_specific_concepts[d] = json_objects_specific[d]
                else:
                    for key, value in current_specific_concepts.items():
                        if key in prev_specific_concepts[d]:
                            if prev_specific_concepts[d][key] != value:
                                prev_specific_concepts[d][key] = value
                        else:
                            prev_specific_concepts[d][key] = value

                #filter the concepts:
                if (si % filter_specific_concepts_frequency_th ==0) and (si !=0):
                    prev_specific_concepts = filter_specific_concept_by_thresh(prev_specific_concepts,domain_sub_datast['train'],model,example_number,n_specific,d,temperature,debug = debug_current,iteration_start=0, iteration_end=si)
                    #save the domain specific concepts dict
                    data_utils.save_text(json.dumps(prev_specific_concepts[d], indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_specific_concepts_dict_prev_Domain_{d}_number_of_exampels_{n_specific}.json'))
                
        
        #filter specific concepts:
        # building the count dictionary for the specific concepts:
        basic_dict = {k: 0 for k in ['0','1', '2', '3']} #labels for the specific concepts
        concepts_dict = {k: copy.deepcopy(basic_dict) for k in prev_specific_concepts[d].keys()} #for each specific concept
        preference_dict = {k:copy.deepcopy(concepts_dict) for k in ['0','1','3']} #for each specific concept

        #filter specific concepts:
        #specific_concepts_dict = {k: 0 for k in prev_specific_concepts[d].keys()}
        #specific_concepts_dict_equal = {k: 0 for k in prev_specific_concepts[d].keys()}
        #specific_concepts_dict_not_equal = {k: 0 for k in prev_specific_concepts[d].keys()}

        #example_count_dict = {k: 1 for k in prev_specific_concepts[d].keys()} #start with 1 to avoid division by zero
        number_of_examples_0 = 0 #For debug
        number_of_examples_1 = 0 #For debug
        number_of_examples_3 = 0 #For debug
        specific_concepts_dict = {k: 0 for k in prev_specific_concepts[d].keys()}
        example_count_dict = {k: 0 for k in prev_specific_concepts[d].keys()}
        index_to_upload_debug_domain_classification = {} 
        index_to_upload_debug_domain_classification['food'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] #For debug
        index_to_upload_debug_domain_classification['legal'] = [] #For debug
        number_of_examples = 0
        minus_one_count = 0 #For debug
        two_count = 0 #For debug
        zero_count = 0 #For debug
        one_count = 0 #For debug
        food_index_to_concepts_llm_as_judge_domain  = []
        legal_index_to_concepts_llm_as_judge_domain  = []
        for j in range(0, si+1):
            print(f" Specific Concepts Classification :: Domain {d} Iteration {j+1} of {si+1}")
            # load the llm as judge dataset
            if j in index_to_upload_debug_domain_classification[d]:
                debug_current = True
            else:
                debug_current = False
            #load the data:
            strart_index = j*n_specific
            if strart_index + n_specific > example_number[d]:
                end_index = example_number[d]
            else:
                end_index = strart_index + n_specific
            d_v = domain_sub_datast['train'][d][strart_index:end_index] 
            json_objects , food_index , legal_index = calculate_concept_importance({d:d_v}, model, temperature ,prev_specific_concepts[d],n_specific ,output_path, debug_current, j, test_flag=False, iteration_end=si+1, food_index_to_concepts_llm_as_judge=food_index_to_concepts_llm_as_judge_domain, legal_index_to_concepts_llm_as_judge=legal_index_to_concepts_llm_as_judge_domain, equal_concepts=False, General = False)
            if json_objects == None:
                print("Error: JSON extraction failed. Please check the text.")
                error_count += 1
                if error_count > 5:
                    print("Too many errors, stopping the process filter concept by threshold.")
                    return None
                continue

            for key, value in json_objects[0]['dataset'].items():
                number_of_examples += 1
                competable_item = data_utils.extract_item_by_original_index({d:d_v}, key)
                preference = competable_item['preference'] 
                if preference == 0:
                    number_of_examples_0 += 1
                    dictionary_to_update = preference_dict['0']
                elif preference == 1:
                    number_of_examples_1 += 1
                    dictionary_to_update = preference_dict['1']
                elif preference == 3:
                    number_of_examples_3 += 1
                    dictionary_to_update = preference_dict['3']
                else:
                    print(f"Error: Invalid preference value {preference} for key {key}.")
                    continue
                for key2, value2 in value.items():
                    if key2 in dictionary_to_update:
                        if value2 == 1:
                            dictionary_to_update[key2]['1'] += 1
                            one_count += 1
                        elif value2 == 3: #For debug
                            dictionary_to_update[key2]['3'] += 1
                            minus_one_count += 1
                        elif value2 == 2: #For debug
                            dictionary_to_update[key2]['2'] += 1
                            two_count += 1  
                        elif value2 == 0: #For debug
                            dictionary_to_update[key2]['0'] += 1
                            zero_count += 1

            print(f"minus one count: {minus_one_count}") #For debug
            print(f"two count: {two_count}") #For debug
            print(f"zero count: {zero_count}") #For debug
            print(f"one count: {one_count}") #For debug
        #normalize the counts:
        for key, value in preference_dict.items():
            # if key is '0' normlaize all values in the iiner dictonries by number_of_examples_0
            # if key is '1' normlaize all values in the iiner dictonries by number_of_examples_1
            # if key is '3' normlaize all values in the iiner dictonries by number_of_examples_3
            if key == '0':
                for key2, value2 in value.items():
                    if number_of_examples_0 > 0:
                        value2['0'] = value2['0'] / number_of_examples_0
                        value2['1'] = value2['1'] / number_of_examples_0
                        value2['2'] = value2['2'] / number_of_examples_0
                        value2['3'] = value2['3'] / number_of_examples_0
            elif key == '1':
                for key2, value2 in value.items():
                    if number_of_examples_1 > 0:
                        value2['0'] = value2['0'] / number_of_examples_1
                        value2['1'] = value2['1'] / number_of_examples_1
                        value2['2'] = value2['2'] / number_of_examples_1
                        value2['3'] = value2['3'] / number_of_examples_1
            elif key == '3':
                for key2, value2 in value.items():
                    if number_of_examples_3 > 0:
                        value2['0'] = value2['0'] / number_of_examples_3
                        value2['1'] = value2['1'] / number_of_examples_3
                        value2['2'] = value2['2'] / number_of_examples_3
                        value2['3'] = value2['3'] / number_of_examples_3

        total_score_per_concept = {k: 0 for k in prev_specific_concepts[d].keys()}
        total_score_per_concept_not_zero = {k: 0 for k in prev_specific_concepts[d].keys()}

        for key, value in preference_dict.items():
            if key == '0':
                number_of_examples_key = number_of_examples_0
            elif key == '1':
                number_of_examples_key = number_of_examples_1
            elif key == '3':
                number_of_examples_key = number_of_examples_3
            for key2, value2 in value.items():
                total_score_per_concept[key2] += value2[key]/number_of_examples*number_of_examples_key
                if key != '0':
                    total_score_per_concept_not_zero[key2] += value2[key]/(number_of_examples_1+number_of_examples_3)*number_of_examples_key
            
        
        

        # keep only the concepts that are above threshold
        threshold = 0.30
        total_score_per_concept = {k: v for k, v in total_score_per_concept.items() if v > threshold}
    
        prev_specific_concepts[d] = {k: v for k, v in prev_specific_concepts[d].items() if (k in total_score_per_concept) }

        #save the domain specific concepts dict
        data_utils.save_text(json.dumps(total_score_per_concept, indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_specific_concepts_dict_Domain_{d}_number_of_exampels_{n_specific}.json'))
        #save the domain specific concepts dict with description
        data_utils.save_text(json.dumps(prev_specific_concepts[d], indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_specific_concepts_dict_prev_Domain_{d}_number_of_exampels_{n_specific}.json'))
        # add the llm as judge per specific concepts
        for value2 in domain_sub_datast['train'][d]:
            original_index = value2['original_index']
            if d == 'food':
                food_index_to_concepts_llm_as_judge_domain = dict(food_index_to_concepts_llm_as_judge_domain)
                value2['Specific Concepts llm as judge'] = food_index_to_concepts_llm_as_judge_domain[str(original_index)]
            elif d == 'legal':
                legal_index_to_concepts_llm_as_judge_domain = dict(legal_index_to_concepts_llm_as_judge_domain)
                value2['Specific Concepts llm as judge'] = legal_index_to_concepts_llm_as_judge_domain[str(original_index)]

        # Test the specific concepts on the test set
        prev_specific_concepts[d], food_index_to_concepts_laj_test, legal_index_to_concepts_laj_test = classify_specific_concepts_on_test_set(prev_specific_concepts[d],model,n_specific,temperature,test_data_dict[d],d)
        for value2 in domain_sub_datast['test'][d]:
            original_index = value2['original_index']
            if d == 'food':
                food_index_to_concepts_laj_test = dict(food_index_to_concepts_laj_test)
                value2['Specific Concepts llm as judge'] = food_index_to_concepts_laj_test[str(original_index)]
            elif d == 'legal':
                legal_index_to_concepts_laj_test = dict(legal_index_to_concepts_laj_test)
                value2['Specific Concepts llm as judge'] = legal_index_to_concepts_laj_test[str(original_index)]
        
        data_utils.save_text(json.dumps(domain_sub_datast['train'][d], indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{d}_FullInfo_train_number_of_exampels_{n_specific}.json'))
        data_utils.save_text(json.dumps(domain_sub_datast['test'][d], indent=2), os.path.join(output_path,f'Model{MODEL_NAME}_Domain_{d}_FullInfo_test_number_of_exampels_{n_specific}.json'))

        #I want to save per domain a json file with the folowing structure:
        # {
        #    train:{
        #     "item0": { 
        #         "User Query": "....",
        #         "Response1": "....",
        #         "Response2": "....",
        #         "LLM as Judge": "-1|1",
        #         "General Concepts_llm_as_judge": {
        #             "Concept 1": "-1|0|1|2",
        #             "Concept 2": "-1|0|1|2",
        #             ...
        #         },
        #         "Specific Concepts_llm_as_judge": {
        #             "Concept 1": "-1|0|1|2",
        #             "Concept 2": "-1|0|1|2",
        #             ...
        #         }
        #     },
        #     "item1": {
        #         "User Query": "....",
        #         "Response1": "....",
        #         "Response2": "....",
        #         "LLM as Judge": "-1|1",
        #         "General Concepts_llm_as_judge": {
        #             "Concept 1": "-1|0|1|2",
        #             "Concept 2": "-1|0|1|2",
        #             ...
        #         },
        #         "Specific Concepts_llm_as_judge": {
        #             "Concept 1": "-1|0|1|2",
        #             "Concept 2": "-1|0|1|2",
        #             ...
        #         }
        #     },
        #    }
        #     ...
        # test:
    #       #         # }
        # save the json file




        


                
            

            
                
                
            



if __name__ == "__main__":
   main()
