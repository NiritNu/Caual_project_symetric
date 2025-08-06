
import random
from utils import prompt_utils, data_utils, json_utils, change_label_utils
import os
import traceback
import json

#CHange it when needed as in the main funcrtion:# Constants
BASE_DIR = '/home/nirit/IBMProject'

PARENT_PROMPT_PATH = os.path.join(BASE_DIR, 'code', 'prompts')
BASE_PROMPT_PATH = [os.path.join(PARENT_PROMPT_PATH, 'LLMasAjudgeGraphPhase.txt'), os.path.join(PARENT_PROMPT_PATH, 'GeneralConceptsClassification.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLabel2minusOne.txt'), os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLAbel2minusOneSimpler.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'ChangeConceptLabel2minusSimplerAgressive.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'GeneralConceptsClassificationNoPreferance.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'LLMasAjudge2ways.txt'),
                    os.path.join(PARENT_PROMPT_PATH, 'GeneralConceptsClassificationNoPreferanceNoTargeConcept.txt')]
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



def ChangeMinusOne(json_objects,GeneralConcepts,SpecificConcepts,TargetConcept, chat_session, temperature, n,domain,debug=False, iteration=0,test=False, trails=1, agressive=False):
    TargetLabel = 3
   
    if test:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_Test_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Test_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne.txt'
    else:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Train_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne.txt'

    # make the path to save in a sub path of current concept
    if not os.path.exists(os.path.join(output_path, TargetConcept)):
        os.makedirs(os.path.join(output_path, TargetConcept))
    path_name = os.path.join(TargetConcept, path_name)

    if not debug:
        if not agressive:
            base_prompt = BASE_PROMPT_PATH[3]
        else:
            base_prompt = BASE_PROMPT_PATH[4]
        prompt= prompt_utils.OneByOnePrompt2(base_prompt,json_objects,domain, GeneralConcepts,SpecificConcepts,TargetConcept, TargetLabel)
        try:
            # Use send_message. The input can be a string or a list of Parts.
            # Using prompt directly as a string is often sufficient.
            response = chat_session.send_message(
                prompt, # Vertex AI expects contents as a list of Parts or strings
                generation_config={"temperature": temperature},
            )
            text = response.text # Access text directly from response object

            #response = model.generate_content(prompt, generation_config={"temperature": temperature})
            #text = response.candidates[0].content.parts[0].text
            #save the response
            data_utils.save_text(text, os.path.join(output_path, path_name))
        except Exception as e:
            print(f"Error during model generation: {e}")
            traceback.print_exc()
            text = "Error during model generation. Please check the logs."
    else:
        text = data_utils.load_text(os.path.join(output_path,path_name))
    return text
def ChangeMinusOneRecursive(json_objects,GeneralConcepts,SpecificConcepts,TargetConcept, model, temperature, n,domain,debug=False, iteration=0,test=False, trails=1, agressive=False):
    TargetLabel = 3
    
    if test:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_recursive_Test_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Test_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne_recursive.txt'
    else:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_recursive_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Train_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne_recursive.txt'

    # make the path to save in a sub path of current concept
    if not os.path.exists(os.path.join(output_path, TargetConcept)):
        os.makedirs(os.path.join(output_path, TargetConcept))
    path_name = os.path.join(TargetConcept, path_name)

    if not debug:
        prompt= prompt_utils.OneByOnePrompt2Recursive(BASE_PROMPT_PATH[3],json_objects,domain, GeneralConcepts,SpecificConcepts,TargetConcept, TargetLabel)
        
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path, path_name))
    else:
        text = data_utils.load_text(os.path.join(output_path,path_name))
    return text 

def ChangeMinusOneRecursive_history(json_objects, current_label_chat,reason_for_label,GeneralConcepts,SpecificConcepts,TargetConcept, chat_session, temperature, n,domain,debug=False, iteration=0,test=False, trails=1, agressive=False):
    TargetLabel = 3

    if test:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_recursive_Test_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Test_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne_recursive.txt'

    else:
        #path_name = f'Model{MODEL_NAME}_ChangeMinusOne_recursive_Domain_{domain}_Key_{TargetConcept}_number_of_examples{n}_attempt_number_{trails}_iteration_{iteration}.txt'
        path_name = f'Model{MODEL_NAME}_Train_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ChangeMinusOne_recursive.txt'

    # make the path to save in a sub path of current concept
    if not os.path.exists(os.path.join(output_path, TargetConcept)):
        os.makedirs(os.path.join(output_path, TargetConcept))
    path_name = os.path.join(TargetConcept, path_name)

    if not debug:
        prompt= prompt_utils.OneByOnePrompt2RecursiveHistoryBased(json_objects,current_label_chat,TargetConcept,reason_for_label)
        
        try:
            # Use send_message. The input can be a string or a list of Parts.
            # Using prompt directly as a string is often sufficient.
            response = chat_session.send_message(
                prompt, # Vertex AI expects contents as a list of Parts or strings
                generation_config={"temperature": temperature},
            )
            text = response.text # Access text directly from response object

            #response = model.generate_content(prompt, generation_config={"temperature": temperature})
            #text = response.candidates[0].content.parts[0].text
            #save the response
            data_utils.save_text(text, os.path.join(output_path, path_name))
        except Exception as e:
            print(f"Error during model generation: {e}")
            traceback.print_exc()
            text = "Error during model recusive history function."
    else:
        text = data_utils.load_text(os.path.join(output_path,path_name))
    return text

def llmAsJudge(data_set, domain,model, temperature, n=2, debug=False, iteration=0, test_flag=False, slice=None, domain_number = 2):
    name_to_save = f'Model{MODEL_NAME}_llmAsJudge_Domain_{domain}_number_of_examples{n}_iteration_{iteration}.txt'
    name_to_save_original = f'Model{MODEL_NAME}_llmAsJudge_Original_Chosen_Domain_{domain}_number_of_examples{n}_iteration_{iteration}.txt'
    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_llmAsJudge_Test_Domain_{domain}_number_of_examples{n}_iteration_{iteration}.txt'
        name_to_save_original = f'Model{MODEL_NAME}_llmAsJudge_Test_Original_Chosen_Domain_{domain}_number_of_examples{n}_iteration_{iteration}.txt'

  

    if not debug:
        prompt, original_chosen = prompt_utils.createPromptLLMasJudgeGraphPhase(BASE_PROMPT_PATH[1],data_set)
        #feed to model stage llm as a judge - #TO DO: add major vote to preferance, as a loop
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path,name_to_save))
        #save the original chosen
        data_utils.save_text(original_chosen, os.path.join(output_path,name_to_save_original))
    else:
        text = data_utils.load_text(os.path.join(output_path,name_to_save)) 
        original_chosen = data_utils.load_text(os.path.join(output_path,name_to_save_original))    
    return text, original_chosen

def llmAsJudge2ways(json_objects,TargetConcept,model, temperature ,n, debug,iteration,iteration_end = None ,test_flag=False, trails = 1,domain = None, direction = 'Chosen first'):

    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_Test_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_llmasjudge2ways_Direction_{direction}.txt'
    else:
        name_to_save = f'Model{MODEL_NAME}_Train_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_llmasjudge2ways_Direction_{direction}.txt'

    # make the path to save in a sub path of current concept
    
    name_to_save = os.path.join(TargetConcept, name_to_save)
    
    if not debug:
        #prompt for the model
        prompt = prompt_utils.createPromptLLMasJudge2Ways(BASE_PROMPT_PATH[6], json_objects[0], direction)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path,name_to_save))
    else:
        text = data_utils.load_text(os.path.join(output_path,name_to_save))
    
    return text

def llmAsJudge2waysBaseData(json_objects, model, temperature, n, debug, iteration, iteration_end=None, test_flag=False, trails=1, domain=None, direction='Chosen first', op = None):
    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_Test_Domain_{domain}_iteration_{iteration}_attempt_number_{trails}_llmasjudge2waysBaseData_Direction_{direction}.txt'
    else:
        name_to_save = f'Model{MODEL_NAME}_Train_Domain_{domain}_iteration_{iteration}_attempt_number_{trails}_llmasjudge2waysBaseData_{direction}.txt'

    if not debug:
        #prompt for the model
        prompt = prompt_utils.createPromptLLMasJudge2WaysBaseData(BASE_PROMPT_PATH[6], json_objects, direction)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(op,name_to_save))
    else:
        text = data_utils.load_text(os.path.join(op,name_to_save))
    
    return text

def llmAsJudgeAvreage(json_objects,TargetConcept,model, temperature ,n, debug,iteration,iteration_end = None ,test_flag=False, trails = 1,domain = None):
    llmAsJudge2ways_chosen = llmAsJudge2ways(json_objects, TargetConcept, model, temperature, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, trails=trails, domain=domain, direction='Chosen first')
    llmAsJudge2ways_chosen = json_utils.check_json_extraction(llmAsJudge2ways_chosen, 1)
    if llmAsJudge2ways_chosen == None:
        error_count += 1
        print("Error: JSON extraction failed. Please check the text.")
        return None
    #json_objects_change_minus1 = json_utils.check_all_items_in_json(json_objects_change_minus1, 1) TO DO : think If I need to add acase for this
    llmAsJudge2ways_rejected = llmAsJudge2ways(json_objects, TargetConcept, model, temperature, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, trails=trails, domain=domain, direction='Rejected first')
    llmAsJudge2ways_rejected = json_utils.check_json_extraction(llmAsJudge2ways_rejected, 1)
    if llmAsJudge2ways_rejected == None:
        error_count += 1
        print("Error: JSON extraction failed. Please check the text.")
        return None
    chosen_prefersnce = [value['preference']for key, value in llmAsJudge2ways_chosen[0].items()]
    rejected_preference = [value['preference']for key, value in llmAsJudge2ways_rejected[0].items()]
    total_preference = []
    for i, item in enumerate(chosen_prefersnce): #going through all examples, usally there is one
        current_chosen = int(item)
        current_rejected = int(rejected_preference[i])
        if (current_chosen == 1) and( current_rejected == 3): #They agree that the first response is better
            total_preference.append(1)
        elif (current_chosen == 3) and( current_rejected == 1): #They agree that the second response is better
            total_preference.append(3)
        else: #They disagree, so we take label it zero
            total_preference.append(0)

    return total_preference

def llmAsJudgeAvreageForBaseData(json_objects, model, temperature, n, debug, iteration, iteration_end=None, test_flag=False, trails=1, domain=None, op=None):
    llmAsJudge2ways_chosen = llmAsJudge2waysBaseData(json_objects, model, temperature, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, trails=trails, domain=domain, direction='Chosen first', op=op)
    llmAsJudge2ways_chosen = json_utils.check_json_extraction(llmAsJudge2ways_chosen, 1)
    if llmAsJudge2ways_chosen == None:
        error_count += 1
        print("Error: JSON extraction failed. Please check the text.")
        return None
    #json_objects_change_minus1 = json_utils.check_all_items_in_json(json_objects_change_minus1, 1) TO DO : think If I need to add acase for this
    llmAsJudge2ways_rejected = llmAsJudge2waysBaseData(json_objects, model, temperature, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, trails=trails, domain=domain, direction='Rejected first', op=op)
    llmAsJudge2ways_rejected = json_utils.check_json_extraction(llmAsJudge2ways_rejected, 1)
    if llmAsJudge2ways_rejected == None:
        error_count += 1
        print("Error: JSON extraction failed. Please check the text.")
        return None
    chosen_prefersnce = [value['preference']for key, value in llmAsJudge2ways_chosen[0].items()]
    rejected_preference = [value['preference']for key, value in llmAsJudge2ways_rejected[0].items()]
    total_preference = []
    for i, item in enumerate(chosen_prefersnce): #going through all examples, usally there is one
        current_chosen = int(item)
        current_rejected = int(rejected_preference[i])
        if (current_chosen == 1) and( current_rejected == 3): #They agree that the first response is better
            total_preference.append(1)
        elif (current_chosen == 3) and( current_rejected == 1): #They agree that the second response is better
            total_preference.append(3)
        else: #They disagree, so we take label it zero
            total_preference.append(0)

    return total_preference
        

 
def calculate_concept_importance(json_objects,TargetConcept,response_direction,model, temperature ,General_concepts, specific_concepts,n, debug,iteration,iteration_end = None ,test_flag=False, mission = 'No_mission"', trails = 1,domain = None):

    if test_flag:
        #name_to_save = f'Model{MODEL_NAME}_ConceptsClassification_Domain_{domain}_Key_{TargetConcept}_Direction_{response_direction}_{mission}_Test_iteration_{iteration}_number_of_examples{n}_trails_{trails}.txt'
        name_to_save = f'Model{MODEL_NAME}_Test_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ConceptsClassification_{mission}_Direction_{response_direction}.txt'

    else:
        #name_to_save = f'Model{MODEL_NAME}_ConceptsClassification__Domain_{domain}_Key_{TargetConcept}_Direction_{response_direction}_{mission}_iteration_{iteration}_out_of_{iteration_end}_number_of_examples{n}_trails_{trails}.txt'
        name_to_save = f'Model{MODEL_NAME}_Train_Domain_{domain}_Key_{TargetConcept}_iteration_{iteration}_attempt_number_{trails}_ConceptsClassification_{mission}_Direction_{response_direction}.txt'

   # make the path to save in a sub path of current concept
    if not os.path.exists(os.path.join(name_to_save, TargetConcept)):
        os.makedirs(os.path.join(name_to_save, TargetConcept))
    name_to_save = os.path.join(TargetConcept, name_to_save)

    if not debug:
        #prompt for the model
        prompt = prompt_utils.createPromptConceptClassificationMinus1NoVisablePreferance(BASE_PROMPT_PATH[5], json_objects, General_concepts, specific_concepts,TargetConcept, direction = response_direction)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(output_path,name_to_save))
    else:
        text = data_utils.load_text(os.path.join(output_path,name_to_save))

    return text

def calculate_concept_importanceBaseData(json_objects,response_direction,model, temperature ,General_concepts, specific_concepts,n, debug,iteration,iteration_end = None ,test_flag=False, trails = 1,domain = None, op=None):

    if test_flag:
        name_to_save = f'Model{MODEL_NAME}_Test_Domain_{domain}_iteration_{iteration}_ConceptsClassification_BaseData_Direction_{response_direction}.txt'

    else:
        name_to_save = f'Model{MODEL_NAME}_Train_Domain_{domain}_iteration_{iteration}_ConceptsClassification_BaseData_Direction_{response_direction}.txt'

    if not debug:
        #prompt for the model
        prompt = prompt_utils.createPromptConceptClassificationMinus1NoVisablePreferanceBaseData(BASE_PROMPT_PATH[7], json_objects, General_concepts, specific_concepts, direction = response_direction)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        text = response.candidates[0].content.parts[0].text
        #save the response
        data_utils.save_text(text, os.path.join(op,name_to_save))
    else:
        text = data_utils.load_text(os.path.join(op,name_to_save))

    return text
def concept_calculation_postprocessing(text,TargetConcept, concept_list, direction):

    json_objects = json_utils.check_json_extraction(text, 1)
    if json_objects == None:
        print("Error: JSON extraction failed. Please check the text.")
        return None, None
    json_objects = json_utils.check_all_items_in_json(json_objects, 2,concept_list)
    reason_for_label = {TargetConcept: json_objects[0][TargetConcept]} if TargetConcept in json_objects[0] else {}
    #change Resonse1 of Response 1 to Chosen Response and Response2 to Rejected Response, since the labling model has no sense of preferance
    for key, value in reason_for_label.items():
        if direction == 'Chosen first':
            value = value.replace('Response1', 'Chosen Response')
            value = value.replace('Response 1', 'Chosen Response')
            value = value.replace('Response2', 'Rejected Response')
            value = value.replace('Response 2', 'Rejected Response')
        else:
            value = value.replace('Response1', 'Rejected Response')
            value = value.replace('Response 1','Rejected Response')
            value = value.replace('Response2', 'Chosen Response')
            value = value.replace('Response 2', 'Chosen Response')
        reason_for_label[key] = value
    return json_objects[0]['dataset'], reason_for_label

def concept_calculation_postprocessingBaseData(text,concept_list):

    json_objects = json_utils.check_json_extraction(text, 1)
    if json_objects == None:
        print("Error: JSON extraction failed. Please check the text.")
        return None, None
    json_objects = json_utils.check_all_items_in_json(json_objects, 2,concept_list)
    
    return json_objects[0]['dataset']
    

def avreage_direction_score_claculation(json_objects, TargetConcept, model, temperature, General_concepts, specific_concepts, concept_list, n, debug=False, iteration=0, iteration_end=None, test_flag=False, mission='No_mission', trails=1, domain=None,agree_count=0, dis_agree_count=0, preferance = 0):
   
    try: # delete this after debugging
        reason_for_label = {}
        text_chosen = calculate_concept_importance(json_objects, TargetConcept, 'Chosen first', model, temperature, General_concepts, specific_concepts, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, mission=mission, trails=trails, domain=domain)
        text_rejected = calculate_concept_importance(json_objects, TargetConcept, 'Rejected first', model, temperature, General_concepts, specific_concepts, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, mission=mission, trails=trails, domain=domain)
        json_objects_chosen, reason_for_label_chosen  = concept_calculation_postprocessing(text_chosen, TargetConcept, concept_list,'Chosen first')
        json_objects_rejected, reason_for_label_rejected  = concept_calculation_postprocessing(text_rejected, TargetConcept,concept_list,'Rejected first')
        #reason_for_label['Chosen'] = reason_for_label_chosen
        #reason_for_label['Rejected'] = reason_for_label_rejected
        if preferance == 0: #no preferance, chose randomly between 1 and 3
            preferance = random.choice([1, 3])
        if preferance == 1:
            reason_for_label = reason_for_label_chosen
        elif preferance == 3:
            reason_for_label = reason_for_label_rejected
    except Exception as e:
        print(f"Error during concept calculation postprocessing: {e}")
        traceback.print_exc()
        return None, None, agree_count, dis_agree_count       
    #initiate average json_object:
    json_objects_avg = {k:{}for k in json_objects_chosen.keys()}
    #average the scores for the chosen and rejected responses To Do
    #check all keys are the same in json_objects:
    # I need to remember that because I changed the oreder of the responses:
    #  label = 1 for chosen is the same as label = 3 for rejected and vicversa
    # label = 0 and label = 2 is the same

    if json_objects_chosen is not None and json_objects_rejected is not None:
        chosen_keys = set(json_objects_chosen.keys())
        rejected_keys = set(json_objects_rejected.keys())
        if chosen_keys != rejected_keys:
            print("Warning: Keys mismatch between json_objects_chosen and json_objects_rejected.")
            print("Keys only in chosen:", chosen_keys - rejected_keys)
            print("Keys only in rejected:", rejected_keys - chosen_keys)
            return None
    for key, value in json_objects_chosen.items(): #going through all exampels
        rejected_match  = json_objects_rejected[key]
        #Check inner dictionary keys:
        if value is not None and rejected_match is not None:
            chosen_keys = set(value.keys())
            rejected_keys = set(rejected_match.keys())
            if chosen_keys != rejected_keys:
                print("Warning: Keys mismatch between json_objects_chosen and json_objects_rejected.")
                return None
        for key2, value2 in value.items():
            rejected_score = rejected_match[key2]
            if (rejected_score == '0') or (value2 == '0'): #if one of the cases checked both responses , so we keep it:
               json_objects_avg[key][key2] = '0'
               if (rejected_score == '0') and (value2 == '0'):
                   agree_count +=1
               else:
                   dis_agree_count +=1
            if rejected_score == '3' and value2 == '3':
                dis_agree_count += 1
                json_objects_avg[key][key2] = '0'
            if rejected_score == '1' and value2 == '1':
                dis_agree_count += 1
                json_objects_avg[key][key2] = '0' 
            if (rejected_score =='2') and (value2=='2'):
                json_objects_avg[key][key2] = '2' 
                agree_count +=1
            if (value2=='3' and rejected_score =='1') or (value2=='2' and rejected_score =='1') or (value2=='3' and rejected_score =='2'):
                json_objects_avg[key][key2] = '3'
                if (value2=='3' and rejected_score =='1'):
                    agree_count +=1
                else:
                    dis_agree_count +=1
            if (value2=='1' and rejected_score =='3') or (value2=='1' and rejected_score =='2') or (value2=='2' and rejected_score =='3'):
                if (value2=='1' and rejected_score =='3'):
                    agree_count+=1
                else:
                    dis_agree_count+=1
                json_objects_avg[key][key2] = '1'

        print(f"Aggrement count: {agree_count}, Disagreement count: {dis_agree_count}")

        #adding llm as a judge for referance:


        return json_objects_avg,reason_for_label, agree_count, dis_agree_count 
    
def avreage_direction_score_claculationBaseData(json_objects, model, temperature, General_concepts, specific_concepts, concept_list, n, debug=False, iteration=0, iteration_end=None, test_flag=False, domain=None,agree_count=0, dis_agree_count=0, preferance = 0, op = None):
   
    text_chosen = calculate_concept_importanceBaseData(json_objects,'Chosen first', model, temperature, General_concepts, specific_concepts, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, domain=domain, op =op)
    text_rejected = calculate_concept_importanceBaseData(json_objects, 'Rejected first', model, temperature, General_concepts, specific_concepts, n, debug, iteration, iteration_end=iteration_end, test_flag=test_flag, domain=domain, op=op)
    json_objects_chosen  = concept_calculation_postprocessingBaseData(text_chosen, concept_list)
    json_objects_rejected  = concept_calculation_postprocessingBaseData(text_rejected,concept_list)
        
    #initiate average json_object:
    json_objects_avg = {k:{}for k in json_objects_chosen.keys()}
    #average the scores for the chosen and rejected responses To Do
    #check all keys are the same in json_objects:
    # I need to remember that because I changed the oreder of the responses:
    #  label = 1 for chosen is the same as label = 3 for rejected and vicversa
    # label = 0 and label = 2 is the same

    if json_objects_chosen is not None and json_objects_rejected is not None:
        chosen_keys = set(json_objects_chosen.keys())
        rejected_keys = set(json_objects_rejected.keys())
        if chosen_keys != rejected_keys:
            print("Warning: Keys mismatch between json_objects_chosen and json_objects_rejected.")
            print("Keys only in chosen:", chosen_keys - rejected_keys)
            print("Keys only in rejected:", rejected_keys - chosen_keys)
            return None
    for key, value in json_objects_chosen.items(): #going through all exampels
        rejected_match  = json_objects_rejected[key]
        #Check inner dictionary keys:
        if value is not None and rejected_match is not None:
            chosen_keys = set(value.keys())
            rejected_keys = set(rejected_match.keys())
            if chosen_keys != rejected_keys:
                print("Warning: Keys mismatch between json_objects_chosen and json_objects_rejected.")
                return None
        for key2, value2 in value.items():
            rejected_score = rejected_match[key2]
            if (rejected_score == '0') or (value2 == '0'): #if one of the cases checked both responses , so we keep it:
               json_objects_avg[key][key2] = '0'
               if (rejected_score == '0') and (value2 == '0'):
                   agree_count +=1
               else:
                   dis_agree_count +=1
            if rejected_score == '3' and value2 == '3':
                dis_agree_count += 1
                json_objects_avg[key][key2] = '0'
            if rejected_score == '1' and value2 == '1':
                dis_agree_count += 1
                json_objects_avg[key][key2] = '0' 
            if (rejected_score =='2') and (value2=='2'):
                json_objects_avg[key][key2] = '2' 
                agree_count +=1
            if (value2=='3' and rejected_score =='1') or (value2=='2' and rejected_score =='1') or (value2=='3' and rejected_score =='2'):
                json_objects_avg[key][key2] = '3'
                if (value2=='3' and rejected_score =='1'):
                    agree_count +=1
                else:
                    dis_agree_count +=1
            if (value2=='1' and rejected_score =='3') or (value2=='1' and rejected_score =='2') or (value2=='2' and rejected_score =='3'):
                if (value2=='1' and rejected_score =='3'):
                    agree_count+=1
                else:
                    dis_agree_count+=1
                json_objects_avg[key][key2] = '1'

        print(f"Aggrement count: {agree_count}, Disagreement count: {dis_agree_count}")

        #adding llm as a judge for referance:


        return json_objects_avg, agree_count, dis_agree_count
    
    

def recursive_change_minus_one_specific_items(batch, concept_list,General_concepts, specific_concepts, d, TargetConcept, 
                                 General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
                                 General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
                                 General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
                                 Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
                                 Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
                                 Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
                                 Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
                                 Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
                                 Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, 
                                 General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, 
                                 General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, 
                                 General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, 
                                 General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,
                                 General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,
                                 model, model_classification, temperature, batch_size, debug=False, iter_number=0, itreration_end=100, trails=1, agressive=False):

    #Note: Here in the recurcive the original label does not have to be -1, it can be 0 or 2 as well
    max_attempts = 3
    max_trails = 5
    if trails > max_trails:
        print("Error: Too many trails occurred. Exiting.")
        return General_concept_dict_0_1, General_concept_dict_0_2, General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,\
              General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,\
              Food_concept_dict_0_1, Food_concept_dict_0_2, Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,\
              Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,\
              Legal_concept_dict_0_1, Legal_concept_dict_0_2, Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,\
              Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
              General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
               General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1
    
    error_count = 0
    no_change_were_made_even_when_needed = 0
    llm_unnecessry_modification = 0
    batch_for_recursion = []
    new_data_to_save = {d: []}
    json_objects_change_minus1 = None
    json_objects = None
    
    while error_count < max_attempts and json_objects_change_minus1 == None:
        try:
            text = ChangeMinusOneRecursive(batch, General_concepts, specific_concepts[d], TargetConcept,model, temperature, domain=d, n=batch_size, debug=debug, iteration=iter_number, trails=trails, agressive=agressive)
            json_objects_change_minus1 = json_utils.check_json_extraction(text, 1)
            if json_objects_change_minus1 == None:
                error_count += 1
                print("Error: JSON extraction failed. Please check the text.")
                if error_count >= max_attempts:
                    print("Max attempts reached. Exiting.")
                    return None
                #check classification of the concepts for the modified labels
            json_objects_change_minus1 = json_utils.check_all_items_in_json(json_objects_change_minus1, 1)
            #make sure that if object deleted than to delet is from the batch as well
            #find the Original Index in all json_objects_change_minus1 and keep on batch only those items
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
                
                
        except:
            error_count += 1
            print("Error: Recursive modificatopn failed. Please check the text.")
    
    error_count = 0
    while error_count < max_attempts and json_objects == None:
        try:
            if not agressive:
                msn = 'minus_one_recursive'
            else:
                msn = 'minus_one_recursive_agressive'
            text = calculate_concept_importance(json_objects_change_minus1,TargetConcept, model_classification, temperature, General_concepts, specific_concepts[d], batch_size, debug=debug, iteration=iter_number, iteration_end=itreration_end, test_flag=False, mission=msn, trails=trails, domain = d)
            json_objects = json_utils.check_json_extraction(text, 1)
            if json_objects == None:
                error_count += 1
                print("Error: JSON extraction failed. Please check the text.")
                if error_count >= max_attempts:
                    print("Max attempts reached. Exiting.")
                    return None
            json_objects = json_utils.check_all_items_in_json(json_objects, 2,concept_list)
        except:
            error_count += 1
            print("Error: recursive modification. Please check the text.")
    
    #check the data extracted:
    for index, item in json_objects[0].items():
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
            if 'Original Index' in b:
                if index_in_batch == str(b['Original Index']):
                    fit_batch2 = b
                    break
            elif 'original_index' in b:
                if index_in_batch == str(b['original_index']):
                    fit_batch2 = b
                    break
        #find the original label of the concept
        if 'Concepts' in fit_batch2:
            original_label = int(fit_batch2['Concepts'][TargetConcept]) 
        elif 'General Concepts llm as judge' in fit_batch2:
            if TargetConcept in fit_batch2['General Concepts llm as judge'].keys():
                original_label = int(fit_batch2['General Concepts llm as judge'][TargetConcept])
            else:
                original_label = int(fit_batch2['Specific Concepts llm as judge'][TargetConcept])
        new_label = int(item[TargetConcept])
        #Case 1: the llm did not change the responses even when it was needed
        if (not fit_batch["ChangeFlag"]) & (original_label != 3): #No change were made but the original label is not the target label
            if new_label == original_label:
                            no_change_were_made_even_when_needed += 1
                            batch_for_recursion.append(fit_batch)
                            continue
        #Case 2" the llm changed the responses even when it was not needed
        if (fit_batch["ChangeFlag"]) & (original_label == 3): #change were made but the original label is the target label
            llm_unnecessry_modification += 1
            continue
        #Case 3: the llm changed the responses and it was needed
        if (fit_batch["ChangeFlag"] or (original_label != new_label)) & (original_label != 3): #change were made and the original label is not the target label
            #Case 3.1" the new label remains the same as the original label
            if new_label == original_label:
                batch_for_recursion.append(fit_batch)
            #case 3.2: the lable changed from original in the opposite direction (from 0 to 1):
            #in this case I dont want to save the change and call the modification agin with the original in batch 
            elif (new_label == 1) and (original_label == 0):
                batch_for_recursion.append(fit_batch2)
            #case 3.3: the new label changed to somthing else then the original label
            else: #I wanto to save what appens even if the label is not the target label
                #TO DO: call the modification agin
                #TO DO: save the new dataset
                #update the normalization factors:
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
                                                                                                                                                                                                        General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,
                                                                                                                                                                                                        General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1)
                                                                             

                for key, value in item.items():
                    if key == TargetConcept:
                        continue
                    #find the original label of the concept
                    original_label_current = int(fit_batch2['Concepts'][key])
                    current_label = int(value)

                    if current_label == original_label_current:
                        continue
                    #extract the right dict to save the changes
                    relevent_dict = change_label_utils.dict_change_extraction(General_concepts, d, original_label, new_label, key,
                                                            General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
                                                            General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
                                                            General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
                                                            Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
                                                            Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
                                                            Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
                                                            Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
                                                            Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
                                                            Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1)
                    
                    if (original_label_current == 0) and (current_label == 1):
                        relevent_dict[key]['0-1'] += 1
                    elif (original_label_current == 0) and (current_label == 2):
                        relevent_dict[key]['0-2'] += 1
                    elif (original_label_current == 0) and (current_label == 3):
                        relevent_dict[key]['0-(-1)'] += 1
                    elif (original_label_current == 1) and (current_label == 0):
                        relevent_dict[key]['1-0'] += 1
                    elif (original_label_current == 1) and (current_label == 2):
                        relevent_dict[key]['1-2'] += 1
                    elif (original_label_current == 1) and (current_label == 3):
                        relevent_dict[key]['1-(-1)'] += 1
                    elif (original_label_current == 2) and (current_label == 0):
                        relevent_dict[key]['2-0'] += 1
                    elif (original_label_current == 2) and (current_label == 1):
                        relevent_dict[key]['2-1'] += 1
                    elif (original_label_current == 2) and (current_label == 3):
                        relevent_dict[key]['2-(-1)'] += 1
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
    
    #save the new dataset
    if len(new_data_to_save[d]) > 0:
        if not agressive:
            with open(os.path.join(new_data_path, f'Model_{model}_new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
                json.dump(new_data_to_save, f, indent=4)
        else:
            with open(os.path.join(new_data_path, f'Model_{model}_new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{trails}__number_of_exampels_{batch_size}_agressive.json'), 'w') as f:
                json.dump(new_data_to_save, f, indent=4)

    # Only make the recursive call if we have batches that need further processing
    if batch_for_recursion:
        # Increment trails for the next recursive call
        next_trails = trails + 1
        return recursive_change_minus_one_specific_items(
            batch_for_recursion,concept_list, General_concepts, specific_concepts, d, TargetConcept, 
            General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
            General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
            General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
            Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
            Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
            Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
            Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
            Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
            Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1,
            General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, 
            General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, 
            General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, 
            General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1, 
            General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,
            model, model_classification, temperature, batch_size, debug=debug, iter_number=iter_number, 
            itreration_end=itreration_end, trails=next_trails, agressive=agressive)
    
    # If there's nothing more to process, return the dictionaries
    return General_concept_dict_0_1, General_concept_dict_0_2, General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,\
              General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,\
              Food_concept_dict_0_1, Food_concept_dict_0_2, Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,\
              Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,\
              Legal_concept_dict_0_1, Legal_concept_dict_0_2, Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,\
              Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
              General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
               General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1

def recursive_change_minus_one_specific_items_history_chat(batch, concept_list,General_concepts, specific_concepts, d, TargetConcept, 
                                 General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
                                 General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
                                 General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
                                 Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
                                 Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
                                 Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
                                 Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
                                 Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
                                 Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, 
                                 General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, 
                                 General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, 
                                 General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, 
                                 General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,
                                 General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,
                                 chat_session, current_label_from_chat,reason_for_label, model_classification, temperature, batch_size, debug=False, iter_number=0, itreration_end=100, trails=1, agressive=False,agree_count=0, dis_agree_count=0):

    #Note: Here in the recurcive the original label does not have to be -1, it can be 0 or 2 as well
    max_attempts = 3
    max_trails = 5
    if trails > max_trails:
        print("Error: Too many trails occurred. Exiting.")
        return General_concept_dict_0_1, General_concept_dict_0_2, General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,\
              General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,\
              Food_concept_dict_0_1, Food_concept_dict_0_2, Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,\
              Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,\
              Legal_concept_dict_0_1, Legal_concept_dict_0_2, Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,\
              Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
              General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
               General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,agree_count, dis_agree_count
    
    error_count = 0
    no_change_were_made_even_when_needed = 0
    llm_unnecessry_modification = 0
    batch_for_recursion = []
    new_data_to_save = {d: []}
    json_objects_change_minus1 = None
    json_objects = None
    
    while error_count < max_attempts and json_objects_change_minus1 == None:
        try:
            text = ChangeMinusOneRecursive_history(batch,current_label_from_chat,reason_for_label, General_concepts, specific_concepts[d], TargetConcept,chat_session, temperature, domain=d, n=batch_size, debug=debug, iteration=iter_number, trails=trails, agressive=agressive)
            json_objects_change_minus1 = json_utils.check_json_extraction(text, 1)
            if json_objects_change_minus1 == None:
                error_count += 1
                print("Error: JSON extraction failed. Please check the text.")
                if error_count >= max_attempts:
                    print("Max attempts reached. Exiting.")
                    return None
                #check classification of the concepts for the modified labels
            json_objects_change_minus1 = json_utils.check_all_items_in_json(json_objects_change_minus1, 1)
            #make sure that if object deleted than to delet is from the batch as well
            #find the Original Index in all json_objects_change_minus1 and keep on batch only those items
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
            #check preferances of the llm for the changes:
            preferance = llmAsJudgeAvreage(json_objects_change_minus1, TargetConcept, model_classification, temperature,batch_size ,debug=debug, iteration=iter_number, iteration_end=itreration_end, test_flag=False, trails=trails, domain=d)  
        
                
                
        except:
            error_count += 1
            print("Error: Recursive modificatopn failed. Please check the text.")
    
    error_count = 0
    new_label_from_chat = []
    while error_count < max_attempts and json_objects == None:
        try:
            if not agressive:
                msn = 'minus_one_recursive'
            else:
                msn = 'minus_one_recursive_agressive'
            json_objects,reason_for_label,agree_count, dis_agree_count = avreage_direction_score_claculation(json_objects_change_minus1, TargetConcept, model_classification, temperature, General_concepts, specific_concepts[d], concept_list, batch_size, debug=debug, iteration=iter_number, iteration_end=itreration_end, test_flag=False,trails=trails, mission=msn,domain=d,agree_count=agree_count, dis_agree_count=dis_agree_count, preferance=preferance[0])
           # text = calculate_concept_importance(json_objects_change_minus1,TargetConcept, model_classification, temperature, General_concepts, specific_concepts[d], batch_size, debug=debug, iteration=iter_number, iteration_end=itreration_end, test_flag=False, mission=msn, trails=trails, domain = d)
           # json_objects = json_utils.check_json_extraction(text, 2)
           # if json_objects == None:
           #     error_count += 1
           #     print("Error: JSON extraction failed. Please check the text.")
           #     if error_count >= max_attempts:
           #         print("Max attempts reached. Exiting.")
           #         return None
            #json_objects = json_utils.check_all_items_in_json(json_objects, 2,concept_list)
           # reason_for_label = {TargetConcept: json_objects[0][TargetConcept]} if TargetConcept in json_objects[0] else {}
            #replace resonse 1 with chosen response, and response 2 with rejected response
           # for key, value in reason_for_label.items():
           #     value = value.replace('Response 1', 'Chosen Response')
           #     value = value.replace('Response1', 'Chosen Response')
           #     value = value.replace('Response 2', 'Rejected Response')
           #     value = value.replace('Response2', 'Rejected Response')
           #     reason_for_label[key] = value

        except:
            error_count += 1
            print("Error: recursive modification. Please check the text.")
    
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
            if 'Original Index' in b:
                if index_in_batch == str(b['Original Index']):
                    fit_batch2 = b
                    break
            elif 'original_index' in b:
                if index_in_batch == str(b['original_index']):
                    fit_batch2 = b
                    break
        #find the original label of the concept
        if 'Concepts' in fit_batch2:
            original_label = int(fit_batch2['Concepts'][TargetConcept]) 
        elif 'General Concepts llm as judge' in fit_batch2:
            if TargetConcept in fit_batch2['General Concepts llm as judge'].keys():
                original_label = int(fit_batch2['General Concepts llm as judge'][TargetConcept])
            else:
                original_label = int(fit_batch2['Specific Concepts llm as judge'][TargetConcept])
        new_label = int(item[TargetConcept])
        new_label_from_chat.append(new_label)
        #Case 1: the llm did not change the responses even when it was needed
        if (not fit_batch["ChangeFlag"]) & (original_label != 3): #No change were made but the original label is not the target label
            if new_label == original_label:
                            no_change_were_made_even_when_needed += 1
                            batch_for_recursion.append(fit_batch)
                            continue
        #Case 2" the llm changed the responses even when it was not needed
        if (fit_batch["ChangeFlag"]) & (original_label == 3): #change were made but the original label is the target label
            llm_unnecessry_modification += 1
            continue
        #Case 3: the llm changed the responses and it was needed
        if (fit_batch["ChangeFlag"] or (original_label != new_label)) & (original_label != 3): #change were made and the original label is not the target label
            #Case 3.1" the new label remains the same as the original label
            if new_label == original_label:
                batch_for_recursion.append(fit_batch)
            #case 3.2: the lable changed from original in the opposite direction (from 0 to 1):
            #in this case I dont want to save the change and call the modification agin with the original in batch 
            elif (new_label == 1) and (original_label == 0):
                batch_for_recursion.append(fit_batch2)
            #case 3.3: the new label changed to somthing else then the original label
            else: #I wanto to save what appens even if the label is not the target label
                #TO DO: call the modification agin
                #TO DO: save the new dataset
                #update the normalization factors:
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
                                                                                                                                                                                                        General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,
                                                                                                                                                                                                        General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1)
                                                                             

                for key, value in item.items():
                    if key == TargetConcept:
                        continue
                    #find the original label of the concept
                    original_label_current = int(fit_batch2['Concepts'][key])
                    current_label = int(value)

                    if current_label == original_label_current:
                        continue
                    #extract the right dict to save the changes
                    relevent_dict = change_label_utils.dict_change_extraction(General_concepts, d, original_label, new_label, key,
                                                            General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
                                                            General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
                                                            General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
                                                            Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
                                                            Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
                                                            Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
                                                            Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
                                                            Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
                                                            Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1)
                    
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
    
    #save the new dataset
    if len(new_data_to_save[d]) > 0:
        if not agressive:
            with open(os.path.join(new_data_path, f'Model_{MODEL_NAME}_new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{trails}_number_of_exampels_{batch_size}.json'), 'w') as f:
                json.dump(new_data_to_save, f, indent=4)
        else:
            with open(os.path.join(new_data_path, f'Model_{MODEL_NAME}_new_data_to_save_{d}_key_{TargetConcept}_itreation_{iter_number}_trail_{trails}__number_of_exampels_{batch_size}_agressive.json'), 'w') as f:
                json.dump(new_data_to_save, f, indent=4)

    # Only make the recursive call if we have batches that need further processing
    if batch_for_recursion:
        # Increment trails for the next recursive call
        next_trails = trails + 1
        return recursive_change_minus_one_specific_items_history_chat(
            batch_for_recursion,concept_list, General_concepts, specific_concepts, d, TargetConcept, 
            General_concept_dict_0_minus1, General_concept_dict_0_1, General_concept_dict_0_2,
            General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,
            General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,
            Food_concept_dict_0_minus1, Food_concept_dict_0_1, Food_concept_dict_0_2,
            Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,
            Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,
            Legal_concept_dict_0_minus1, Legal_concept_dict_0_1, Legal_concept_dict_0_2,
            Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,
            Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1,
            General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, 
            General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, 
            General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, 
            General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1, 
            General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1,
            chat_session,new_label_from_chat,reason_for_label, model_classification, temperature, batch_size, debug=debug, iter_number=iter_number, 
            itreration_end=itreration_end, trails=next_trails, agressive=agressive,agree_count=agree_count, dis_agree_count=dis_agree_count)
    
    # If there's nothing more to process, return the dictionaries
    return General_concept_dict_0_1, General_concept_dict_0_2, General_concept_dict_1_minus1, General_concept_dict_1_0, General_concept_dict_1_2,\
              General_concept_dict_2_minus1, General_concept_dict_2_0, General_concept_dict_2_1,\
              Food_concept_dict_0_1, Food_concept_dict_0_2, Food_concept_dict_1_minus1, Food_concept_dict_1_0, Food_concept_dict_1_2,\
              Food_concept_dict_2_minus1, Food_concept_dict_2_0, Food_concept_dict_2_1,\
              Legal_concept_dict_0_1, Legal_concept_dict_0_2, Legal_concept_dict_1_minus1, Legal_concept_dict_1_0, Legal_concept_dict_1_2,\
              Legal_concept_dict_2_minus1, Legal_concept_dict_2_0, Legal_concept_dict_2_1, General_concept_norm_0_1, Food_concept_norm_0_1, Legal_concept_norm_0_1, General_concept_norm_0_2, Food_concept_norm_0_2, Legal_concept_norm_0_2, \
              General_concept_norm_1_minus1, Food_concept_norm_1_minus1, Legal_concept_norm_1_minus1, General_concept_norm_1_0, Food_concept_norm_1_0, Legal_concept_norm_1_0, \
               General_concept_norm_1_2, Food_concept_norm_1_2, Legal_concept_norm_1_2, General_concept_norm_2_minus1, Food_concept_norm_2_minus1, Legal_concept_norm_2_minus1, \
                General_concept_norm_2_0, Food_concept_norm_2_0, Legal_concept_norm_2_0, General_concept_norm_2_1, Food_concept_norm_2_1, Legal_concept_norm_2_1,\
                General_concept_norm_0_minus1, Food_concept_norm_0_minus1, Legal_concept_norm_0_minus1, agree_count, dis_agree_count