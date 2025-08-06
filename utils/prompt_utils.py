import json
import random


def load_prompt(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def create_prompt2(data, base_prompt_path,general_concepts = 2, specific_concepts = 1, previuos_concepts = None):
    base_prompt = load_prompt(base_prompt_path)
    ## add the data to prompt
    # find the place to insert the data
    start = base_prompt.find('Insert Data')
    if previuos_concepts is not None:
        end = base_prompt.find('Previos output json file')
    else:
        end = base_prompt.find('Task')
    tail = base_prompt[end:]
    base_prompt = base_prompt[:start]
    for i, example in enumerate(data):
        base_prompt += f"\n\nExample {i+1}:\n"
        base_prompt += f"User Query: {example['Q']}\n"
        base_prompt += f"Chosen: {example['C']}\n"
        base_prompt += f"Rejected: {example['R']}\n"
        base_prompt += f"Domain: {example['D']}\n"
    base_prompt += tail
    if previuos_concepts is not None:
        # Insert the previus concepts
        start = base_prompt.find('Insert Json')
        end = base_prompt.find('Task:')
        tail = base_prompt[end:]
        base_prompt = base_prompt[:start]
        base_prompt += previuos_concepts
        base_prompt += tail
    else:
        # Insert number of general concepts and specific concepts
        base_prompt = base_prompt.replace('B#', general_concepts)
        base_prompt = base_prompt.replace('S#', specific_concepts)

    return base_prompt
  

def create_prompt(data, base_prompt_path, general_concepts='10', specific_concepts='5',previuos_concepts = None):
    base_prompt = load_prompt(base_prompt_path)
    ## add the data to prompt
    for i, example in enumerate(data):
        base_prompt += f"\n\nExample {i+1}:\n"
        
        base_prompt += f"User Query: {example['User Query']}\n"
        base_prompt += f"Chosen Response: {example['Chosen Response']}\n"
        base_prompt += f"Rejected Response: {example['Rejected Response']}\n"
        base_prompt += f"Domain: {example['Domain']}\n"

    # Insert number of general concepts and specific concepts
    base_prompt = base_prompt.replace('B#', general_concepts)
    base_prompt = base_prompt.replace('S#', specific_concepts)

    ## add previuos concepts in json format:
    if previuos_concepts is not None:
        base_prompt += '\n\ncurrent list of concepts:\n'
        base_prompt += '\n\njson\n'
        base_prompt += previuos_concepts

    return base_prompt

def create_promptFused(base_prompt_path, general_concepts='10', specific_concepts='5'):
    base_prompt = load_prompt(base_prompt_path)
    # Insert number of general concepts and specific concepts
    base_prompt = base_prompt.replace('B#', general_concepts)
    base_prompt = base_prompt.replace('S#', specific_concepts)
    return base_prompt

def create_promptBaseDiffConceptNumber(base_prompt_path, general_concepts='10', specific_concepts='5'):
    base_prompt = load_prompt(base_prompt_path)
    # Insert number of general concepts and specific concepts
    base_prompt = base_prompt.replace('B#', general_concepts)
    base_prompt = base_prompt.replace('S#', specific_concepts)
    return base_prompt

def createPromptLLMasJudge(base_prompt_path, data, direction='Chosen first'):
    base_prompt = load_prompt(base_prompt_path)
    #Helf of the times I want the chose response to be response1 and the other half to be response2
    ## add the data to prompt
    for i, example in enumerate(data):
        base_prompt += f"\n\nExample {i+1}:\n"
        base_prompt += f"User Query: {example['User Query']}\n"
        if direction == 'Chosen first':
            base_prompt += f"Response1: {example['Chosen Response']}\n"
            base_prompt += f"Response2: {example['Rejected Response']}\n"
        else:
            base_prompt += f"Response1: {example['Rejected Response']}\n"
            base_prompt += f"Response2: {example['Chosen Response']}\n"
        base_prompt += f"Domain: {example['Domain']}\n"
        base_prompt += f"Index: {example['Index']}\n"
        base_prompt += f"Original_index: {example['Original_index']}\n"

    return base_prompt

def createPromptExtractGeneral(base_prompt_path,base_prompt_equal_path, json_file, concepts_number):
    base_prompt = load_prompt(base_prompt_path)
    base_prompt2 = load_prompt(base_prompt_path)
    base_prompt_equal1 = load_prompt(base_prompt_equal_path)
    base_prompt_equal2 = load_prompt(base_prompt_equal_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    exmple_number_equal = 0
    for key,value in json_file.items():
        for value2 in value:
            if value2['preference'] == 0: #equal
                base_prompt_equal1 += f"\n\nExample {exmple_number_equal}:\n"
                base_prompt_equal1 += f"User Query: {value2['user_query']}\n"
                base_prompt_equal2 += f"\n\nExample {exmple_number_equal}:\n"
                base_prompt_equal2 += f"User Query: {value2['user_query']}\n"
                base_prompt_equal1 += f"Response1: {value2['response1']}\n"
                base_prompt_equal2 += f"Response2: {value2['response2']}\n"
                base_prompt_equal1 += f"Response2: {value2['response2']}\n"
                base_prompt_equal2 += f"Response1: {value2['response1']}\n"
                base_prompt_equal1 += f"Domain: {key}\n"
                base_prompt_equal2 += f"Domain: {key}\n"
                exmple_number_equal += 1

            else: #not equal
                base_prompt += f"\n\nExample {exmple_number}:\n"
                base_prompt2 += f"\n\nExample {exmple_number}:\n"
                base_prompt += f"User Query: {value2['user_query']}\n"
                base_prompt2 += f"User Query: {value2['user_query']}\n"
                base_prompt += f"Response1: {value2['response1']}\n"
                base_prompt += f"Response2: {value2['response2']}\n"
                base_prompt2 += f"Response1: {value2['response2']}\n"
                base_prompt2 += f"Response2: {value2['response1']}\n"
                preference = value2['preference']
                if preference == 1:
                    base_prompt += f"Chosen: Response1\n"
                    base_prompt2 += f"Chosen: Response2\n"
                elif preference == 3:
                    base_prompt += f"Chosen: Response2\n"
                    base_prompt2 += f"Chosen: Response1\n" 
                base_prompt += f"Domain: {key}\n"
                base_prompt2 += f"Domain: {key}\n"
                exmple_number += 1
    ## add the number of general concepts
    base_prompt = base_prompt.replace('*B#*', concepts_number)
    base_prompt2 = base_prompt2.replace('*B#*', concepts_number)
    base_prompt_equal1 = base_prompt_equal1.replace('*B#*', concepts_number)
    base_prompt_equal2 = base_prompt_equal2.replace('*B#*', concepts_number)

    return base_prompt, base_prompt2, base_prompt_equal1, base_prompt_equal2

def createPromptAddGeneral(base_prompt_path, json_file, concepts_number):
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    for key,value in json_file[0].items():
        for value2 in value:
            base_prompt += f"\n\nExample {exmple_number}:\n"
            base_prompt += f"User Query: {value2['user_query']}\n"
            preference = value2['preference']
            if preference == 1:
                base_prompt += f"Chosen Response: {value2['response1']}\n"
                base_prompt += f"Rejected Response: {value2['response2']}\n"
            else:
                base_prompt += f"Chosen Response: {value2['response2']}\n"
                base_prompt += f"Rejected Response: {value2['response1']}\n"
            base_prompt += f"Domain: {key}\n"
            exmple_number += 1
   ## add the number of general concepts
    base_prompt = base_prompt.replace('"B#"', str(concepts_number))

    return base_prompt

def createPromptGeneralConceptClassification(base_prompt_path, json_objects, concepts_list, General = True):
    base_prompt = load_prompt(base_prompt_path)
    base_prompt2 = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    #for key,value in json_objects.items():
    #    for value2 in value:
    base_prompt += f"\n\nExample {exmple_number}:\n"
    base_prompt2 += f"\n\nExample {exmple_number}:\n"
    base_prompt += f"Original Index: {json_objects['original_index']}\n"
    base_prompt2 += f"Original Index: {json_objects['original_index']}\n"
    base_prompt += f"User Query: {json_objects['user_query']}\n"
    base_prompt2 += f"User Query: {json_objects['user_query']}\n"
    preferance = json_objects['preference']
    if preferance == 0: #
        #randomly choose which response to show first
        if random.choice([True, False]):
            base_prompt += f"Response1: {json_objects['response1']}\n"
            base_prompt += f"Response2: {json_objects['response2']}\n"
            base_prompt2 += f"Response1: {json_objects['response2']}\n"
            base_prompt2 += f"Response2: {json_objects['response1']}\n"

        else:
            base_prompt += f"Response1: {json_objects['response2']}\n"
            base_prompt += f"Response2: {json_objects['response1']}\n"
            base_prompt2 += f"Response1: {json_objects['response1']}\n"
            base_prompt2 += f"Response2: {json_objects['response2']}\n"
            
    else:
        chosen_first = random.choice([True, False])

        if chosen_first: #Response1 is the chosen response
            
            if preferance == 1:
                base_prompt += f"Response1: {json_objects['response1']}\n"
                base_prompt += f"Response2: {json_objects['response2']}\n"
                base_prompt2 += f"Response1: {json_objects['response2']}\n"
                base_prompt2 += f"Response2: {json_objects['response1']}\n"
                
            elif preferance == 3:
                base_prompt += f"Response1: {json_objects['response2']}\n"
                base_prompt += f"Response2: {json_objects['response1']}\n"
                base_prompt2 += f"Response1: {json_objects['response1']}\n"
                base_prompt2 += f"Response2: {json_objects['response2']}\n"

        else: #Response2 is the chosen response
            if preferance == 1:
                base_prompt += f"Response1: {json_objects['response2']}\n"
                base_prompt += f"Response2: {json_objects['response1']}\n"
                base_prompt2 += f"Response1: {json_objects['response1']}\n"
                base_prompt2 += f"Response2: {json_objects['response2']}\n"
                
            elif preferance == 3:
                base_prompt += f"Response1: {json_objects['response1']}\n"
                base_prompt += f"Response2: {json_objects['response2']}\n"
                base_prompt2 += f"Response1: {json_objects['response2']}\n"
                base_prompt2 += f"Response2: {json_objects['response1']}\n"
    exmple_number += 1
   ## add  general concepts
    if General:
        base_prompt += f"\n\nGeneral Concepts:\n"
        base_prompt2 += f"\n\nGeneral Concepts:\n"
    else:
        base_prompt += f"\n\nSpecific Concepts:\n"
        base_prompt2 += f"\n\nSpecific Concepts:\n"
    base_prompt += json.dumps(concepts_list, indent=2) 
    base_prompt2 += json.dumps(concepts_list, indent=2) 

    return base_prompt, base_prompt2
def createPromptSpecificConceptClassification(base_prompt_path, json_objects, concepts_list):
    base_prompt = load_prompt(base_prompt_path)
    base_prompt2 = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    #for key,value in json_objects.items():
    #    for value2 in value:
    base_prompt += f"\n\nExample {exmple_number}:\n"
    base_prompt2 += f"\n\nExample {exmple_number}:\n"
    base_prompt += f"Original Index: {json_objects['original_index']}\n"
    base_prompt2 += f"Original Index: {json_objects['original_index']}\n"
    base_prompt += f"User Query: {json_objects['user_query']}\n"
    base_prompt2 += f"User Query: {json_objects['user_query']}\n"
    preferance = json_objects['preference']
    if preferance == 0: #
        #randomly choose which response to show first
        if random.choice([True, False]):
            base_prompt += f"Response1: {json_objects['response1']}\n"
            base_prompt += f"Response2: {json_objects['response2']}\n"
            base_prompt2 += f"Response1: {json_objects['response2']}\n"
            base_prompt2 += f"Response2: {json_objects['response1']}\n"

        else:
            base_prompt += f"Response1: {json_objects['response2']}\n"
            base_prompt += f"Response2: {json_objects['response1']}\n"
            base_prompt2 += f"Response1: {json_objects['response1']}\n"
            base_prompt2 += f"Response2: {json_objects['response2']}\n"
            
    else:
        chosen_first = random.choice([True, False])

        if chosen_first: #Response1 is the chosen response
            
            if preferance == 1:
                base_prompt += f"Response1: {json_objects['response1']}\n"
                base_prompt += f"Response2: {json_objects['response2']}\n"
                base_prompt2 += f"Response1: {json_objects['response2']}\n"
                base_prompt2 += f"Response2: {json_objects['response1']}\n"
                
            elif preferance == 3:
                base_prompt += f"Response1: {json_objects['response2']}\n"
                base_prompt += f"Response2: {json_objects['response1']}\n"
                base_prompt2 += f"Response1: {json_objects['response1']}\n"
                base_prompt2 += f"Response2: {json_objects['response2']}\n"

        else: #Response2 is the chosen response
            if preferance == 1:
                base_prompt += f"Response1: {json_objects['response2']}\n"
                base_prompt += f"Response2: {json_objects['response1']}\n"
                base_prompt2 += f"Response1: {json_objects['response1']}\n"
                base_prompt2 += f"Response2: {json_objects['response2']}\n"
                
            elif preferance == 3:
                base_prompt += f"Response1: {json_objects['response1']}\n"
                base_prompt += f"Response2: {json_objects['response2']}\n"
                base_prompt2 += f"Response1: {json_objects['response2']}\n"
                base_prompt2 += f"Response2: {json_objects['response1']}\n"
    exmple_number += 1
   ## add  general concepts
    base_prompt += f"\n\nSpecific Concepts:\n"
    base_prompt2 += f"\n\nSpecific Concepts:\n"
    base_prompt += json.dumps(concepts_list, indent=2) 
    base_prompt2 += json.dumps(concepts_list, indent=2)

    return base_prompt, base_prompt2

def createPromptExtractSpecific(base_prompt_path, json_file, concepts_number):
    base_prompt = load_prompt(base_prompt_path)
    base_prompt2 = load_prompt(base_prompt_path)
    #base_prompt_equal1 = load_prompt(base_prompt_equal_path)
    #base_prompt_equal2 = load_prompt(base_prompt_equal_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    #exmple_number_equal = 0
    for key,value in json_file.items():
        for value2 in value:
            if value2['preference'] == 0: #equal
                '''base_prompt_equal1 += f"\n\nExample {exmple_number_equal}:\n"
                base_prompt_equal1 += f"User Query: {value2['user_query']}\n"
                base_prompt_equal2 += f"\n\nExample {exmple_number_equal}:\n"
                base_prompt_equal2 += f"User Query: {value2['user_query']}\n"
                base_prompt_equal1 += f"Response1: {value2['response1']}\n"
                base_prompt_equal2 += f"Response2: {value2['response2']}\n"
                base_prompt_equal1 += f"Response2: {value2['response2']}\n"
                base_prompt_equal2 += f"Response1: {value2['response1']}\n"
                base_prompt_equal1 += f"Domain: {key}\n"
                base_prompt_equal2 += f"Domain: {key}\n"
                exmple_number_equal += 1'''
                continue #skip equal examples, we don't want to extract specific concepts from them

            else: #not equal
                base_prompt += f"\n\nExample {exmple_number}:\n"
                base_prompt2 += f"\n\nExample {exmple_number}:\n"
                base_prompt += f"User Query: {value2['user_query']}\n"
                base_prompt2 += f"User Query: {value2['user_query']}\n"
                base_prompt += f"Response1: {value2['response1']}\n"
                base_prompt += f"Response2: {value2['response2']}\n"
                base_prompt2 += f"Response1: {value2['response2']}\n"
                base_prompt2 += f"Response2: {value2['response1']}\n"
                preference = value2['preference']
                if preference == 1:
                    base_prompt += f"Chosen: Response1\n"
                    base_prompt2 += f"Chosen: Response2\n"
                elif preference == 3:
                    base_prompt += f"Chosen: Response2\n"
                    base_prompt2 += f"Chosen: Response1\n" 
                base_prompt += f"Domain: {key}\n"
                base_prompt2 += f"Domain: {key}\n"
                exmple_number += 1
    ## add the number of general concepts
    base_prompt = base_prompt.replace('*B#*', concepts_number)
    base_prompt2 = base_prompt2.replace('*B#*', concepts_number)
    #base_prompt_equal1 = base_prompt_equal1.replace('*B#*', concepts_number)
    #base_prompt_equal2 = base_prompt_equal2.replace('*B#*', concepts_number)

    return base_prompt, base_prompt2#, base_prompt_equal1, base_prompt_equal2
    

def createPromptFilterConcepts(base_prompt_path, concepts_dictionry):
    base_prompt = load_prompt(base_prompt_path)
    base_prompt += f"\n\nConcepts to group:\n"
    for key, value in concepts_dictionry.items():
        base_prompt += f"{key}: {value}\n"
    return base_prompt

def All1Prompt(base_prompt_path, json_objects, domain, GeneralConcepts, SpecificConcepts):
    base_prompt = load_prompt(base_prompt_path)
    ## add the data to prompt
    for i, example in enumerate(json_objects):
        base_prompt += f"\n\nExample {i+1}:\n"
        base_prompt += f"Original Index: {example['original_index']}\n"
        base_prompt += f"User Query: {example['user_query']}\n"
        preferance = example['preference']
        if preferance == 1:
            base_prompt += f"Chosen: {example['response1']}\n"
            base_prompt += f"Rejected: {example['response2']}\n"
        else:
            base_prompt += f"Chosen: {example['response2']}\n"
            base_prompt += f"Rejected: {example['response1']}\n"
        base_prompt += f"Domain: {domain}\n"
        #adding the General concept, by extracrting the keys from the dictionary:
        base_prompt += f"Concepts and thier labels: \n"
        for key, value in example['General Concepts llm as judge'].items():
            base_prompt += f"{key}: {value}\n"
        #adding the Specific concept, by extracrting the keys from the dictionary:
        for key, value in example['Specific Concepts llm as judge'].items():
            base_prompt += f"{key}: {value}\n"

    base_prompt += f"\n\nConcepts descreption:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"
    
    return base_prompt

def createPromptLLMasJudgeGraphPhase(base_prompt_path, data):
    base_prompt = load_prompt(base_prompt_path)
    #Helf of the times I want the chose response to be response1 and the other half to be response2
    random_choice = random.choice([0, 1]) #it ok all the batch have the same random choice, but all the data is still different
    original_chosen_response = []
    ## add the data to prompt
    for i, example in enumerate(data):
        base_prompt += f"\n\nExample {i+1}:\n"
        base_prompt += f"User Query: {example['User Query']}\n"
        if random_choice == 0:
            base_prompt += f"Response1: {example['New Chosen Response']}\n"
            base_prompt += f"Response2: {example['New Rejected Response']}\n"
            original_chosen_response.append(1)
        else:
            base_prompt += f"Response1: {example['New Rejected Response']}\n"
            base_prompt += f"Response2: {example['New Chosen Response']}\n"
            original_chosen_response.append(2)
        base_prompt += f"Domain: {example['Domain']}\n"
        base_prompt += f"Original_index: {example['Original Index']}\n"

    #convert the list of original chosen response to a string
    original_chosen_response = ', '.join(map(str, original_chosen_response))

    return base_prompt, original_chosen_response

def createPromptLLMasJudge2Ways(base_prompt_path, json_objects, response_direction):
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for key, value in json_objects.items():
        base_prompt += f"\n\nExample {exmple_number}:\n"
        base_prompt += f"Original Index: {value['Original Index']}\n"
        base_prompt += f"User Query: {value['User Query']}\n"
        if response_direction == 'Chosen first':
            base_prompt += f"Response1: {value['Chosen Response']}\n"
            base_prompt += f"Response2: {value['Rejected Response']}\n"  
        else: #Rejected is Response1
            base_prompt += f"Response1: {value['Rejected Response']}\n"
            base_prompt += f"Response2: {value['Chosen Response']}\n"  
       
        exmple_number += 1
  
    return base_prompt

def createPromptLLMasJudge2WaysBaseData(base_prompt_path, json_objects, response_direction):
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for value in json_objects:
        base_prompt += f"\n\nExample {exmple_number}:\n"
        base_prompt += f"Original Index: {value['original_index']}\n"
        base_prompt += f"User Query: {value['user_query']}\n"
        prefereance = value['preference']
        if prefereance == 1: #response1 is the chosen response
            if response_direction == 'Chosen first':
                base_prompt += f"Response1: {value['response1']}\n"
                base_prompt += f"Response2: {value['response2']}\n"  
            else: #Rejected is Response1
                base_prompt += f"Response1: {value['response2']}\n"
                base_prompt += f"Response2: {value['response1']}\n"
        else: #response2 is the chosen response
            if response_direction == 'Chosen first':
                base_prompt += f"Response1: {value['response2']}\n"
                base_prompt += f"Response2: {value['response1']}\n"  
            else: #Rejected is Response1
                base_prompt += f"Response1: {value['response1']}\n"
                base_prompt += f"Response2: {value['response2']}\n"
       
        exmple_number += 1
  
    return base_prompt


def createPromptConceptClassificationALL1(base_prompt_path, json_objects, GeneralConcepts, SpecificConcepts):
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for value in json_objects:
        ## randomlly assign chosen and rejected response to be the first or second to present
        random_choice = random.choice([0, 1])
        
        base_prompt += f"\n\nExample {exmple_number}:\n"
        base_prompt += f"Original Index: {value['Original Index']}\n"
        base_prompt += f"User Query: {value['User Query']}\n"
        if random_choice == 0:
            base_prompt += f"Chosen Response: {value['New Chosen Response']}\n"
            base_prompt += f"Rejected Response: {value['New Rejected Response']}\n"    
        else:
            base_prompt += f"Rejected Response: {value['New Rejected Response']}\n"
            base_prompt += f"Chosen Response: {value['New Chosen Response']}\n"  
            
        exmple_number += 1
   ## add  general concepts
    base_prompt += f"\n\nConcepts:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"

    return base_prompt

def createPromptConceptClassificationMinus1(base_prompt_path, json_objects, GeneralConcepts, SpecificConcepts):
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for key, value in json_objects[0].items():
        ## randomlly assign chosen and rejected response to be the first or second to present
        random_choice = random.choice([0, 1])
        
        base_prompt += f"\n\nExample {exmple_number}:\n"
        if 'preference' in value:
            base_prompt += f"Original Index: {value['original_index']}\n"
            base_prompt += f"User Query: {value['user_query']}\n"
            preferance = value['preference']
            if random_choice == 0:
                if preferance == 1:
                    base_prompt += f"Chosen Response: {value['response1']}\n"
                    base_prompt += f"Rejected Response: {value['response2']}\n"
                else:
                    base_prompt += f"Rejected Response: {value['response1']}\n"
                    base_prompt += f"Chosen Response: {value['response2']}\n"
            else:
                if preferance == 1:
                    base_prompt += f"Rejected Response: {value['response2']}\n"
                    base_prompt += f"Chosen Response: {value['response1']}\n"  
                else:
                    base_prompt += f"Chosen Response: {value['response2']}\n"
                    base_prompt += f"Rejected Response: {value['response1']}\n"
        else:    
            base_prompt += f"Original Index: {value['Original Index']}\n"
            base_prompt += f"User Query: {value['User Query']}\n"
            if random_choice == 0:
                base_prompt += f"Chosen Response: {value['Chosen Response']}\n"
                base_prompt += f"Rejected Response: {value['Rejected Response']}\n"    
            else:
                base_prompt += f"Rejected Response: {value['Rejected Response']}\n"
                base_prompt += f"Chosen Response: {value['Chosen Response']}\n"  
            
        exmple_number += 1
   ## add  general concepts
    base_prompt += f"\n\nConcepts:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"

    return base_prompt
def createPromptConceptClassificationMinus1NoVisablePreferance(base_prompt_path, json_objects, GeneralConcepts, SpecificConcepts,TargetConcepts,direction = 'Chosen first'):
    #This function is similar to createPromptConceptClassificationMinus1 but does not show the preference in the prompt.
    #Note that the chosen responses is always response1 and the rejected response is always response2.
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for key, value in json_objects[0].items():
        
        base_prompt += f"\n\nExample {exmple_number}:\n"
        if 'preference' in value:
            base_prompt += f"Original Index: {value['original_index']}\n"
            base_prompt += f"User Query: {value['user_query']}\n"
            if direction == 'Chosen first':
                preferance = value['preference']
                if preferance == 1:
                    base_prompt += f"Response1: {value['response1']}\n"
                    base_prompt += f"Response2: {value['response2']}\n"
                else:
                    base_prompt += f"Response1: {value['response2']}\n"
                    base_prompt += f"Response2: {value['response1']}\n"
            else: #Rejected is Response1
                preferance = value['preference']
                if preferance == 1:
                    base_prompt += f"Response1: {value['response2']}\n"
                    base_prompt += f"Response2: {value['response1']}\n"
                else:
                    base_prompt += f"Response1: {value['response1']}\n"
                    base_prompt += f"Response2: {value['response2']}\n"
            
        else:    
            base_prompt += f"Original Index: {value['Original Index']}\n"
            base_prompt += f"User Query: {value['User Query']}\n"
            if direction == 'Chosen first':
                base_prompt += f"Response1: {value['Chosen Response']}\n"
                base_prompt += f"Response2: {value['Rejected Response']}\n"   
            else: #Rejected is Response1
                base_prompt += f"Response1: {value['Rejected Response']}\n"
                base_prompt += f"Response2: {value['Chosen Response']}\n" 
            
            
        exmple_number += 1
   ## add  general concepts
    base_prompt += f"\n\nConcepts:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"

    base_prompt = base_prompt.replace('*TargetConcepts*',TargetConcepts)

    
    

    return base_prompt

def createPromptConceptClassificationMinus1NoVisablePreferanceBaseData(base_prompt_path, json_objects,GeneralConcepts,SpecificConcepts,direction = 'Chosen first'):
    #This function is similar to createPromptConceptClassificationMinus1 but does not show the preference in the prompt.
    #Note that the chosen responses is always response1 and the rejected response is always response2.
    base_prompt = load_prompt(base_prompt_path)
    #go through the json file and add the data to the prompt
    exmple_number = 0
    
    for value in json_objects:
        
        base_prompt += f"\n\nExample {exmple_number}:\n"
        if 'preference' in value:
            base_prompt += f"Original Index: {value['original_index']}\n"
            base_prompt += f"User Query: {value['user_query']}\n"
            if direction == 'Chosen first':
                preferance = value['preference']
                if preferance == 1:
                    base_prompt += f"Response1: {value['response1']}\n"
                    base_prompt += f"Response2: {value['response2']}\n"
                else:
                    base_prompt += f"Response1: {value['response2']}\n"
                    base_prompt += f"Response2: {value['response1']}\n"
            else: #Rejected is Response1
                preferance = value['preference']
                if preferance == 1:
                    base_prompt += f"Response1: {value['response2']}\n"
                    base_prompt += f"Response2: {value['response1']}\n"
                else:
                    base_prompt += f"Response1: {value['response1']}\n"
                    base_prompt += f"Response2: {value['response2']}\n"
            
        else:    
            base_prompt += f"Original Index: {value['Original Index']}\n"
            base_prompt += f"User Query: {value['User Query']}\n"
            if direction == 'Chosen first':
                base_prompt += f"Response1: {value['Chosen Response']}\n"
                base_prompt += f"Response2: {value['Rejected Response']}\n"   
            else: #Rejected is Response1
                base_prompt += f"Response1: {value['Rejected Response']}\n"
                base_prompt += f"Response2: {value['Chosen Response']}\n" 
            
            
        exmple_number += 1
   
   ## add  general concepts
    base_prompt += f"\n\nConcepts:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"


    return base_prompt

def OneByOnePrompt(base_prompt_path,json_objects,domain, GeneralConcepts,SpecificConcepts,TargetConcept, TargetLabel):
    base_prompt = load_prompt(base_prompt_path)
    ## add the data to prompt
    for i, example in enumerate(json_objects):
        base_prompt += f"\n\nExample Item{i+1}:\n"
        base_prompt += f"Original Index: {example['original_index']}\n"
        base_prompt += f"User Query: {example['user_query']}\n"
        preferance = example['preference']
        if preferance == 1:
            base_prompt += f"Original Chosen Response: {example['response1']}\n"
            base_prompt += f"Original Rejected Response: {example['response2']}\n"
        else:
            base_prompt += f"Original Chosen Response: {example['response2']}\n"
            base_prompt += f"Original Rejected Response: {example['response1']}\n"
        base_prompt += f"Domain: {domain}\n"
        #adding the General concept, by extracrting the keys from the dictionary:
        base_prompt += f"Original Concept Labels: \n"
        for key, value in example['General Concepts llm as judge'].items():
            base_prompt += f"{key}: {value}\n"
        #adding the Specific concept, by extracrting the keys from the dictionary:
        for key, value in example['Specific Concepts llm as judge'].items():
            base_prompt += f"{key}: {value}\n"

    #adding the concepts description
    base_prompt += f"\n\nConcepts Descriptions:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"
    
    #adding the Target concept and label
    base_prompt += f"\n\nTarget Concept Name: {TargetConcept}\n"
    base_prompt += f"Target Concept Label: {TargetLabel}\n"

    return base_prompt

def OneByOnePrompt2(base_prompt_path,json_objects,domain, GeneralConcepts,SpecificConcepts,TargetConcept, TargetLabel):
    base_prompt = load_prompt(base_prompt_path)

    #adding the concepts description
    base_prompt += f"\n\nConcepts Descriptions:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
        if key == TargetConcept:
                targerDescription = value
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"
        if key == TargetConcept:
                targerDescription = value
    
    base_prompt += f"\n\nThe dataset:\n"
    ## add the data to prompt
    for i, example in enumerate(json_objects):
        base_prompt += f"\n\nItem{i+1}:\n"
        base_prompt += f"Domain: {domain}\n"
        base_prompt += f"Original Index: {example['original_index']}\n"
        base_prompt += f"User Query: {example['user_query']}\n"
        preferance = example['preference']
        if preferance == 1:
            base_prompt += f"Original Chosen Response: {example['response1']}\n"
            base_prompt += f"Original Rejected Response: {example['response2']}\n"
        else:
            base_prompt += f"Original Chosen Response: {example['response2']}\n"
            base_prompt += f"Original Rejected Response: {example['response1']}\n"
        #adding the General concept, by extracrting the keys from the dictionary:
        base_prompt += f"Original Concept Labels: \n"
        for key, value in example['General Concepts llm as judge'].items():
            if key in GeneralConcepts:
                base_prompt += f"{key}: {value}\n"   
        #adding the Specific concept, by extracrting the keys from the dictionary:
        for key, value in example['Specific Concepts llm as judge'].items():
            base_prompt += f"{key}: {value}\n"
            
    #replace the Target concept and label
    base_prompt = base_prompt.replace('*Target Concept*', TargetConcept)
    base_prompt = base_prompt.replace('*Target Label*', str(TargetLabel))
    base_prompt = base_prompt.replace('*Concept descreption*',targerDescription)
    
    return base_prompt

def OneByOnePrompt2Recursive(base_prompt_path,json_objects,domain, GeneralConcepts,SpecificConcepts,TargetConcept, TargetLabel):
    base_prompt = load_prompt(base_prompt_path)

    #adding the concepts description
    base_prompt += f"\n\nConcepts Descriptions:\n"
    for key, value in GeneralConcepts.items():
        base_prompt += f"{key}: {value}\n"
    for key, value in SpecificConcepts.items():
        base_prompt += f"{key}: {value}\n"
    
    base_prompt += f"\n\nThe dataset:\n"
    ## add the data to prompt
    for i, example in enumerate(json_objects):
        base_prompt += f"\n\nItem{i+1}:\n"
        base_prompt += f"Domain: {domain}\n"
        #if preference key exist: then the format is different
        if 'preference' in example:
            base_prompt += f"Original Index: {example['original_index']}\n"
            base_prompt += f"User Query: {example['user_query']}\n"
            preferance = example['preference']
            if preferance == 1:
                base_prompt += f"Chosen Response: {example['response1']}\n"
                base_prompt += f"Rejected Response: {example['response2']}\n"
            else:
                base_prompt += f"Chosen Response: {example['response2']}\n"
                base_prompt += f"Rejected Response: {example['response1']}\n"

            base_prompt += f"Original Concept Labels: \n"
            for key, value in example['General Concepts llm as judge'].items():
                base_prompt += f"{key}: {value}\n"
            #adding the Specific concept, by extracrting the keys from the dictionary:
            for key, value in example['Specific Concepts llm as judge'].items():
                base_prompt += f"{key}: {value}\n"

         
        
        else:
            base_prompt += f"Original Index: {example['Original Index']}\n"
            base_prompt += f"User Query: {example['User Query']}\n"
            base_prompt += f"Chosen Response: {example['Chosen Response']}\n"
            base_prompt += f"Rejected Response: {example['Rejected Response']}\n"
        
            #adding the General concept, by extracrting the keys from the dictionary:
            base_prompt += f"Original Concept Labels: \n"
            for key, value in example['Concepts'].items():
                base_prompt += f"{key}: {value}\n"
        

    #replace the Target concept and label
    base_prompt = base_prompt.replace('*Target Concept*', TargetConcept)
    base_prompt = base_prompt.replace('*Target Label*', str(TargetLabel))
    return base_prompt

def OneByOnePrompt2RecursiveHistoryBased(json_objects, current_label_chat, TargetConcept,reason_for_label):
    # Start the prompt by clearly stating the purpose: providing feedback on responses.
    #check if all items have a label of '3' dont do a thimg
    if all(label == 3 for label in current_label_chat):
        return "All items have been successfully labeled with '3'. No further modifications are needed."
    base_prompt = "Here's the feedback I've received on the responses you modified, broken down by each item:"

    good_example_flag = False # Flag to track if any item received a '3' label

    # Loop through each item to provide specific feedback
    for i, example in enumerate(json_objects):
        # Only provide detailed feedback for items that are NOT labeled '3'
        if current_label_chat[i] != 3:
            base_prompt += f"\n\nFeedback Item {i+1}:"
            base_prompt += f"\nOriginal Index: {example['Original Index']}"
            base_prompt += f"\nLabel Received: {current_label_chat[i]} – meaning: "

            if current_label_chat[i] == 1:
                # Label 1: Chosen response is better for the concept
                base_prompt += f"The chosen response better represents the concept '{TargetConcept}'."
            elif current_label_chat[i] == 2:
                # Label 2: Target concept is missing from both responses
                base_prompt += f"The concept '{TargetConcept}' is not present in any response."
            elif current_label_chat[i] == 0:
                # Label 0: Both responses represent the concept equally (or equally poorly)
                base_prompt += f"The concept '{TargetConcept}' is equally represented in both the chosen and rejected responses."
            base_prompt += 'This is the reason the llm gave for the target concept label:'
            base_prompt += f"\n The reason of one of the llms that labeled the concept: {reason_for_label[TargetConcept]}"
        else:
            # If a label '3' is found, set the flag. We'll list these separately later.
            good_example_flag = True

    # Main instruction for items that need further modification
    base_prompt += "\n\nBased on this feedback and the reason the llm gave for the labeling, the responses for the items listed above are NOT yet meeting the desired outcome. Please continue to refine their chosen and rejected responses, ensuring you strictly follow the original instructions."

    # Crucial guidelines for how to perform the modifications
    base_prompt += "\n\nCrucial Guidelines for Modifications:"
    base_prompt += "\n- Always base your changes on the *original* chosen and rejected responses to preserve their core meaning and context."
    base_prompt += "\n- Continue modifying the responses until they are *better* than the original ones, but *never* change the fundamental meaning of either response."
    base_prompt += "\n- Ensure both the chosen and rejected responses remain *coherent* and *consistent* with their original context." # 
    base_prompt +="\n- add to the reasoninig , how you consider the reason the llm gave for the labeling in you modifications."

    # If there were any items with a '3' label, list them as completed.
    if good_example_flag:
        base_prompt += "\n\nGood news! For the following items, the concept is now correctly represented in the rejected response. These items are complete and require no further changes:"
        # Loop again specifically to list items with label '3'
        for k, example_k in enumerate(json_objects):
            if current_label_chat[k] == 3:
                base_prompt += f"\n- Item {k+1} (Original Index: {example_k['Original Index']}): Label Received: {current_label_chat[k]} – meaning: The concept '{TargetConcept}' is now correctly represented in the rejected response."

    return base_prompt