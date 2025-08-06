import json
import os
import verification

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    
def average_json_objects(json_objects_chosen, json_objects_rejected, case = 'llmAsJudge'):
    #TO DO: Implement the logic to average the JSON objects.
    if case == 'llmAsJudge':
        for d, items in json_objects_chosen.items():
            for i, item in enumerate(items):
                if item['original_index'] != json_objects_rejected[d][i]['original_index']:
                    print(f"Error: Original index mismatch in dataset {d} for item {i}.")
                    return None
                preference_chosen = item['preference']
                preference_rejected = json_objects_rejected[d][i]['preference']
                if preference_chosen == 1 and preference_rejected == 3:
                    item['preference'] = 1
                elif preference_chosen == 3 and preference_rejected == 1:
                    item['preference'] = 3
                elif preference_chosen == 3 and preference_rejected == 3:
                    item['preference'] = 0
                elif preference_chosen == 1 and preference_rejected == 1:
                    item['preference'] = 0
                else:
                    print(f"Error: Preference mismatch in dataset {d} for item {i}.")
                    return None
        return [json_objects_chosen]

        
    elif case == 'concept_classification':
        # Implement the averaging logic for another case
        data_chosen = json_objects_chosen['dataset']
        data_rejected = json_objects_rejected['dataset']
        for key in data_chosen.keys():
            if key not in data_rejected:
                print(f"Error: Key {key} not found in rejected dataset.")
                return None
            for concept, label_chosen in data_chosen[key].items():
                label_chosen = int(label_chosen)
                if concept not in data_rejected[key]:
                    #check maybe there is unwanted backspace btween two letters in the name of the concept
                    #remove backspace from concept if exist:
                    old_concept = concept
                    concept_rejected = concept.replace(" ", "")
                    concept = old_concept
                    #remove backspace from the data_rejected[key] keys:
                    data_rejected[key] = {k.replace(" ", ""): v for k, v in data_rejected[key].items()}
                    if concept_rejected not in data_rejected[key]:
                        print(f"Error: Concept {concept} not found in rejected dataset for key {key}.")
                        return None
                else:
                    concept_rejected = concept
                label_rejected = int(data_rejected[key][concept_rejected])
                # Average the labels (assuming they are numeric)
                if (label_chosen == 0) or (label_rejected == 0) or (label_chosen == 1 and label_rejected == 1) or (label_chosen == 3 and label_rejected == 3):
                    # If label is 0, we consider it as present in both cases, if the label is 1 or 3 in both cases, we consider it as present in both cases.
                    data_chosen[key][concept] = 0
                elif (label_chosen == 1 and label_rejected == 3) or (label_chosen == 1 and label_rejected == 2) or (label_chosen == 2 and label_rejected == 3):
                    data_chosen[key][concept] = 1
                elif (label_chosen == 3 and label_rejected == 1) or (label_chosen == 2 and label_rejected == 1) or (label_chosen == 3 and label_rejected == 2):
                    data_chosen[key][concept] = 3
                elif (label_chosen == 2) and (label_rejected == 2):
                    data_chosen[key][concept] = 2
        return [json_objects_chosen]

def find_matching_brace(text, start_index):
    """
    Finds the index of the matching closing brace for a given opening brace.
    """
    brace_count = 1
    for i in range(start_index + 1, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        if brace_count == 0:
            return i
    return -1  # No matching brace found.

def check_json_extraction(text, number_of_expected_json):
    json_objects,_ = json_extract(text)
    #json_objects = json_extract_version2_with_ENDJSON(text)

    #check if the json objects are valid, if not manuallu exytact the json objects
    if not verification.number_of_jsons_extracted(json_objects,number_of_expected_json): # not two json files...somethong went wrong and I want to extract the text using "json", and END JSON
        text = fix_json_files_if_extrcted_failed(text)
        if text == None:
            print("Error: JSON extraction failed. Please check the text.")
            return None
        elif text == 'json\n' + '\nEND JSON\n': #No json were extracted and there is nothiing to fix, need a new iteration
            print("Error: No JSON objects were found in the text.")
            return None
        else:    
            json_objects,_ = json_extract(text)
    
    return json_objects

def extract_json_from_text(text,output_path ,json_file_name_per_example, json_file_name_per_concept, save_json = False):
    """
    Extracts two JSON objects from a text file and saves them as separate JSON files.

    Args:
        text(str): str content of the text file.
        output_path (str): Path to save the JSON files.
        json_file_name_per_example (str): Name of the JSON file for per example data.
        json_file_name_per_concept (str): Name of the JSON file for per concept data.
    """
    try:
        text_split = text.split('json\n') # if there are 3 jason and it is written in the right order, then the first one is text the second is per example and the thried is the concepts
        if len(text_split) == 3:
            #json files are inside the text and capsulated by {}, so I will extract it:
            #extract the first json file:
            item_concepts_json_str_len = len(text_split[1].split('END JASON'))
            item_concepts_json_str = text_split[1].split('END JASON')[0]
            if item_concepts_json_str_len ==1: #Delete the last part of the string The "END JSON" part
                item_concepts_json_str = item_concepts_json_str.replace('END JASON', '').strip()
                
            concept_data_json_str_len = len(text_split[2].split('END JSON'))
            concept_data_json_str = text_split[2].split('END JASON')[0]
            if concept_data_json_str_len ==1:
                concept_data_json_str = concept_data_json_str.replace('END JASON', '').strip()
            #check if the json files are valid:
            is_valid_json(item_concepts_json_str)
            is_valid_json(concept_data_json_str)
            #save the json files:
            if save_json:
                item_concepts = json.loads(item_concepts_json_str)
                concept_data = json.loads(concept_data_json_str) 
            
                with open(os.path.join(output_path,json_file_name_per_example), 'w') as item_concepts_file:
                    json.dump(item_concepts, item_concepts_file, indent=4)
                with open(os.path.join(output_path,json_file_name_per_concept), 'w') as concept_data_file:
                    json.dump(concept_data, concept_data_file, indent=4)
                #This part verify it savied the json files correctly, if not then it will raise an error, now I will delete it:
                
            return item_concepts_json_str, concept_data_json_str
        else:
            print("Error: Expected 3 JSON objects in the text.")
            return None, None

    except:
        print("Error: Could not extract JSON from text.")
        return None, None
    
def json_extract(text):
    json_objects = []
    end_index_vec = []
    start_index = 0
    while True:
        start_index = text.find('{', start_index)
        if start_index == -1:
            break  # No more JSON objects found.

        end_index = find_matching_brace(text, start_index)
        if end_index == -1:
            break #malformed json
        end_index_vec.append(end_index) #append the end index to the list, so I can use it later.

        json_string = text[start_index:end_index + 1]
        try:
            parsed_json = json.loads(json_string)
            json_objects.append(parsed_json)
            start_index = end_index + 1  # Move past the current JSON object.
        except json.JSONDecodeError:
            start_index += 1 #move one character and try again.
    return json_objects, end_index_vec 

def json_extract_version2_with_ENDJSON(text):
    """Extract JSON objects that are followed by 'END JSON' marker."""
    json_objects = []
    start_index = 0
    
    while True:
        start_index = text.find('{', start_index)
        if start_index == -1:
            break  # No more JSON objects found.

        end_index = find_matching_brace(text, start_index)
        if end_index == -1:
            break  # Malformed JSON

        json_string = text[start_index:end_index + 1]
        
        # Check if "END JSON" appears after the closing brace
        remaining_text = text[end_index + 1:].strip()
        if not remaining_text.startswith('END JSON'):
            start_index += 1  # Move one character and try again
            continue
        
        try:
            parsed_json = json.loads(json_string)
            json_objects.append(parsed_json)
            # Move past the "END JSON" marker
            end_json_pos = text.find('END JSON', end_index + 1)
            start_index = end_json_pos + len('END JSON')
        except json.JSONDecodeError:
            start_index += 1  # Move one character and try again
    
    return json_objects

import json
import re

import json
import re

def fix_json_string(json_string):
    """
    Attempts to fix JSON strings with missing commas, braces, extra data, or missing double quotes.

    Args:
        json_string: The potentially invalid JSON string.

    Returns:
        The fixed JSON string, or None if fixing fails.
    """
    try:
        json.loads(json_string)  # Check if it's already valid
        return json_string
    except json.JSONDecodeError as e:
        error_message = str(e)
        if "Expecting ',' delimiter" in error_message:
            # Attempt to fix missing commas
            try:
                # Use regex to find potential missing comma locations
                fixed_string = re.sub(r'}(?=\s*"|\s*{|\s*\[|\s*\w)', '},', json_string)
                fixed_string = re.sub(r'](?=\s*"|\s*{|\s*\[|\s*\w)', '],', fixed_string)
                fixed_string = re.sub(r'"(?=\s*"|\s*{|\s*\[|\s*\w)', '",', fixed_string)
                fixed_string = re.sub(r'(?<![\[\{\s])(?=\s*"|\s*{|\s*\[|\s*\w)(?<!:)', ',', fixed_string)
                #remove trailing commas.
                fixed_string = re.sub(r',(\s*[\]\}])', r'\1', fixed_string)
                json.loads(fixed_string) # check if it is valid after fix.
                return fixed_string, error_message
            except (json.JSONDecodeError, re.error) as e:
                error_message = str(e)
                return fixed_string, error_message  # Fixing commas failed
        elif "Expecting '}'" in error_message or "Expecting ']'" in error_message:
            # Attempt to fix missing braces or brackets
            brace_count = json_string.count('{') - json_string.count('}')
            bracket_count = json_string.count('[') - json_string.count(']')

            fixed_string = json_string
            if brace_count > 0:
                fixed_string += '}' * brace_count
            elif brace_count < 0:
                fixed_string = '\{'*abs(brace_count) + fixed_string

            if bracket_count > 0:
                fixed_string += ']' * bracket_count
            elif bracket_count < 0:
                fixed_string = '\['*abs(bracket_count) + fixed_string

            try:
                json.loads(fixed_string)
                return fixed_string, error_message
            except json.JSONDecodeError as e:
                error_message = str(e)
                return fixed_string, error_message #fixing braces failed
        elif "Extra data" in error_message:
            # Attempt to fix extra data error
            try:
                # Find the location of the extra data
                match = re.search(r"\(char (\d+)\)", error_message)
                if match:
                    char_index = int(match.group(1))
                    # Truncate the string at the extra data point
                    fixed_string = json_string[:char_index]
                    try:
                        json.loads(fixed_string)
                        return fixed_string, error_message
                    except json.JSONDecodeError as e:
                        error_message = str(e)
                        # Try to remove the last comma, or bracket before the extra data.
                        fixed_string = re.sub(r',(\s*)$', '', fixed_string)
                        fixed_string = re.sub(r'\](\s*)$', '', fixed_string)
                        fixed_string = re.sub(r'}(\s*)$', '', fixed_string)

                        json.loads(fixed_string)
                        return fixed_string, error_message

                else:
                    return None, error_message
            except (json.JSONDecodeError, re.error, ValueError):
                return fixed_string, error_message
        elif "Expecting property name enclosed in double quotes" in error_message:
            # Attempt to fix missing double quotes around property names
            try:
                # Use regex to find potential missing quotes around keys
                fixed_string = re.sub(r'([{\s,])(\w+)(?=:)', r'\1"\2"', json_string) #improved regex.
                json.loads(fixed_string)
                return fixed_string, error_message
            except (json.JSONDecodeError, re.error):
                try:
                    # attempt to fix single quotes
                    fixed_string = re.sub(r"'(.*?)'", r'"\1"', json_string)
                    json.loads(fixed_string)
                    return fixed_string, error_message
                except (json.JSONDecodeError, re.error) as e:
                    error_message = str(e)
                    try:
                        # try to fix both single and missing double quotes
                        fixed_string = re.sub(r"'(.*?)'", r'"\1"', json_string)
                        fixed_string = re.sub(r'([{\s,])(\w+)(?=:)', r'\1"\2"', fixed_string)
                        json.loads(fixed_string)
                        return fixed_string, error_message
                    except(json.JSONDecodeError, re.error) as e:
                        error_message = str(e)
                        # 4. Fix trailing commas
                        try:
                            fixed_string = re.sub(r",(\s*[\]}])", r"\1", json_string)
                            json.loads(fixed_string)
                            return fixed_string, error_message
                        except json.JSONDecodeError as e:
                            error_message = str(e)
                            return fixed_string, error_message
        elif "Invalid \\escape" in error_message:
            # Attempt to fix invalid backslash escapes
            try:
                # Replace backslash followed by a non-valid escape char with just the char
                # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
                # Regex: Find a backslash NOT followed by one of the valid chars or u
                # Use a negative lookahead (?!)
                # We need to be careful not to replace valid escapes like \n or \"
                # Let's try replacing '\\' followed by anything not in the allowed set [\"\\/bfnrtu]
                # This might be too aggressive, alternative is removing the '\'

                # Approach 1: Remove the offending backslash if it's not escaping a valid character
                # This looks for '\' followed by anything *not* in the set {\, ", /, b, f, n, r, t, u}
                # and replaces it with just the character that followed.
                fixed_string = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_string)

                json.loads(fixed_string)
                return fixed_string, error_message
            except (json.JSONDecodeError, re.error) as e_fix:
                 # Fixing escapes failed
                return fixed_string, error_message # Return the attempted fix and original error
            
        elif "Unterminated string" in error_message:
            # Attempt to fix unterminated string by adding a closing quote at the end
            # This is a basic fix and might place the quote incorrectly.
            try:
                fixed_string = json_string + '"'
                json.loads(fixed_string)
                return fixed_string, error_message
            except (json.JSONDecodeError, re.error) as e_fix:
                 # Adding quote failed to fix or caused another error
                return fixed_string, error_message # Return the attempted fix and original error

        elif "Invalid control character" in error_message:
            # New error handling: Attempt to fix invalid control characters
            try:
                # 1. Escape common allowed control characters that appear unescaped
                fixed_string = json_string.replace('\t', '\\t') # Replace literal tab with escaped tab
                #fixed_string = fixed_string.replace('\n', '\\n') # Replace literal newline with escaped newline
                #fixed_string = fixed_string.replace('\r', '\\r') # Replace literal carriage return with escaped carriage return

                # 2. Remove other problematic Unicode control and format characters that JSON dislikes
                # Covers C0 controls (excluding \t, \n, \r), C1 controls, Zero Width/Format chars,
                # Line/Paragraph Separators, BOM.
                fixed_string = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', fixed_string)
                
                json.loads(fixed_string) # Check if valid after this fix
                return fixed_string, error_message
            except (json.JSONDecodeError, re.error) as e_fix:
                # If removing them still doesn't fix it, or causes a new regex error.
                return fixed_string, error_message
        else:
            return None, error_message  # Unhandled error
        

def out_loop_fix_string(string_to_fix, max_iterations = 10):
    """
    Attempts to fix a JSON string iteratively until it becomes valid or reaches the max iterations. In case of more then 1 error in the string, it will
    try to fix it in a loop, until it becomes valid or the max iterations is reached.

    Args:
        string_to_fix: The potentially invalid JSON string.
        max_iterations: The maximum number of iterations to attempt fixing.

    Returns:
        The fixed JSON string, or None if fixing fails.
    """
    prev_error_message = None
    fixed_string = string_to_fix
    for _ in range(max_iterations):
        fixed_string, current_error = fix_json_string(fixed_string)
        if fixed_string is None:
            return None
        try:
            json.loads(fixed_string)  # Check if it's already valid
            return fixed_string
        except json.JSONDecodeError:
            pass
        if current_error == prev_error_message: #Im in endless loop, so I will break the loop.
            return None
        prev_error_message = current_error 
        
       
    return None  # Max iterations reached without success

def fix_json_files_if_extrcted_failed(text):

    # before continuing with this funciton, I need to chek the case where the 'END JSON; string is part of the json and not in the end:
    #This is the case where the extraction succed , but the END JSON is part of the json object, so I need to remove it from the json object and add it to the text.   
    #TODO for now it is to only 1 json object, but I will change it to multiple json objects later.
    try:
        json_texts = []
        json_objects, end_loc = json_extract(text)
        if json_objects is not None and len(json_objects) > 0: 
            #find if END JSON is a json object key or value, and if it is, dekete it and add 'END JSON' to the text right after the json object
            
            for i, json_object in enumerate(json_objects):
                #Case 1: Json object wa extracted and 'END JSON; is akey inside the json object, so I need to remove it from the json object and add it to the end of the text, after the json obkects.
                if 'END JSON' in json_object:
                    #remove the 'END JSON' from the json object
                    del json_objects[i]['END JSON']
                    #add 'END JSON' to the text right after the json object
                    #text = text.replace(json.dumps(json_object), json.dumps(json_object) + '\nEND JSON\n')
                    json_texts.append(json.dumps(json_object))
                #Case 2: Json object was extracted and 'END JSON' is just not in the text, so I will add it to the end of the text
                elif 'END JSON' not in text[end_loc[i] + 1:]:
                    json_texts.append(json.dumps(json_object))
            text = 'json\n' + '\nEND JSON\n'.join(json_texts) + '\nEND JSON\n'
            return text #if json objects are valid, then I will return the text with the '
        #Case
        
    except: #if extraction failed, I will try to fix the json files
        json_texts = []
        tmp = text.split('json\n') # json files strart with token "json" and end with "END JSON"
        for k in range(1, len(tmp)): #thw fisrt part is not json file it is what before that
            json_texts.append(tmp[k].split('END JSON')[0])
        for l, json_text in enumerate(json_texts):
            if not is_valid_json(json_text):
                json_texts[l] = out_loop_fix_string(json_text,200)
                    #After traing to fix json , if I git None the I could not fix it and run new iteration, else it is fixed and I need to concat all texts and run json_extract again
        if None in json_texts:
            print("Error: Could not fix JSON files. Please check the input.")
            return None
        else:
            #concat all texts
            text = 'json\n' + '\nEND JSON\n'.join(json_texts) + '\nEND JSON\n'
        return text

def check_all_items_in_json(json_objects, case, concept_list_to_check = None):

    #Case 1: all object should contain: Original Index, User Query, Chosen Response, Rejected Response, ChangeFlag
    #if not delete that item from the json object
    if case == 1:
        a = json_objects[0]
        # Create a list of keys to delete to avoid modifying during iteration
        keys_to_delete = []
        for key, value in a.items():
            if 'Original Index' not in value or 'User Query' not in value or 'Chosen Response' not in value or 'Rejected Response' not in value or 'ChangeFlag' not in value:
                keys_to_delete.append(key)
        
        # Delete the keys after iteration
        for key in keys_to_delete:
            del json_objects[0][key]
                
    if case == 2: #Check all items are in concept list, if not, delete that item from the json object, 
        #also check that the parent keys are: 1.dataset, 2 valid concept name 
        keys_to_delete = []
        #check if 'dataset' is in the json object, if not abort:
        if 'dataset' not in json_objects[0]:
            print("Error: 'dataset' key not found in the JSON object.")
            return None
        for k in json_objects[0].keys():
            if not ((k == 'dataset') or (k in concept_list_to_check)):
                print(f"Key {k} is not in the concept list or not 'dataset'. Deleting item.")
                keys_to_delete.append(k)
        # Delete the keys after iteration
        for k in keys_to_delete:
            del json_objects[0][k]
                  
        a = json_objects[0]
        a = a['dataset']
        for key in list(a.keys()):
            value = a[key]
            #if key == domain and value is domain_name (str), delete the item
            if key == 'domain' and isinstance(value, str):
                if value not in concept_list_to_check:
                    del json_objects[0][key]
                continue
            concepts_to_delete = []
            for c in list(value.keys()):
                if c not in concept_list_to_check:
                    concepts_to_delete.append(c)
            # Delete the concepts after iteration
            for c in concepts_to_delete:
                del json_objects[0]['dataset'][key][c]

    if case == 3: #check the main key is 'dataset' 
        if 'dataset' not in json_objects[0]:
            print("Error: 'dataset' key not found in the JSON object.")
            return None
        

    # TODO if case == 3: #json object is a dictonart with target concept in the key and a string in the value:
    #    a = json_objects[0]
        

                
    return json_objects
