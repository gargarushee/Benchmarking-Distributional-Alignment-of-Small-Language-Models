import pandas as pd
import numpy as np
import random, json, os


# constants
dem_group_to_dem_mapping = {'NONE': ['Democrat'], 
                            'POLPARTY': ['Democrat', 'Republican'],
                            'SEX': ['Male', 'Female'],
                            'RACE': ['Black', 'White'] ,
                            'globalvalues': ['0', '1', '2'] 
                            }

options={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R"}


# helper functions
def get_q_IDs_opinionqa(wave="Pew_American_Trends_Panel_disagreement_100"):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    path = open(data_path + wave + '/info.csv')
    df = pd.read_csv(path)
    waves = np.array(df['survey'])
    q_IDs = np.array(df['key'])
    return q_IDs, waves

def get_qIDS(data_path, wave):
    f = open(data_path + wave + '/question_similarity.json')
    data = json.load(f)
    return list(data.keys())        

# Can be removed as we have generic function for dataset   
def get_ICL_qIDs(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_500", 
    demographic_group="POLPARTY", 
    demographic="Democrat",
):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    f = open(data_path + wave + '/question_similarity_top10.json')

    question_similarity_top10 = json.load(f)
    top10 = question_similarity_top10[q_ID]
    icl_data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    # score these ICL qIDs based on similarity in distribution to q_ID
    # first separate the qIDs with the equivalent num MC and not equivalent num MC 
    easy, easy_hard = [], []
    n_qID_MC = len(icl_data[q_ID]['MC_options'])
    for icl_qID in top10: 
        if n_qID_MC == len(icl_data[icl_qID]['MC_options']): easy.append(icl_qID)
        else: easy_hard.append(icl_qID)
    
    if len(easy_hard)==5:
        return easy_hard

    # if len(easy)<len(easy_hard): There are too many that have different MC options
    # Since we want to keep the more challenging ones in easy_hard, move the textually similar ones to easy 
    # the IDs in easy_hard are ranked from most textually similar to least 
    if len(easy)<len(easy_hard):
        num_to_transfer = len(easy_hard) - 5
        easy = easy + easy_hard[:num_to_transfer]
        easy_hard = easy_hard[num_to_transfer:]

    # if len(easy)>len(easy_hard): split based on similarity to ground truth distribution 
    if len(easy)>len(easy_hard):
        num_to_transfer = len(easy) - 5
        distrib_dist = []
        q_ID_values = np.array(list(icl_data[q_ID][demographic].values()))/np.sum(list(icl_data[q_ID][demographic].values()))
        for easy_qID in easy: 
            icl_values = []
            # calculate distributional difference
            for MC_option in icl_data[easy_qID]['MC_options']:
                if MC_option in icl_data[easy_qID][demographic].keys(): 
                    icl_values.append(icl_data[easy_qID][demographic][MC_option])
                else: icl_values.append(0)
                
            icl_values = np.array(icl_values)/np.sum(icl_values)
            try: distrib_dist.append(total_variation(icl_values, q_ID_values))
            except: distrib_dist.append(1)
        sorted_pairs = sorted(zip(distrib_dist, easy))
        # order qIDs based on smallest to largest distribution difference
        sorted_list_qIDs= [pair[1] for pair in sorted_pairs]

        # keep the ones in easy that have the lowest distributional differences 
        easy = sorted_list_qIDs[:5]
        easy_hard = easy_hard + sorted_list_qIDs[5:]

    if len(easy_hard)==5:
        return easy_hard

    return easy_hard

# Get all in context learning question ids for datasets - opinionqa, nytimes, global_values
def get_ICL_qIDs(q_ID, wave, demographic_group, demographic, dataset='opinionqa'):

    if dataset=='opinionqa': data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    else: data_path = '{}/{}'.format(os.getcwd(), dataset)

    icl_data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
        
    if dataset=='opinionqa':
        f = open(data_path + wave + '/question_similarity_top10.json')

        question_similarity_top10 = json.load(f)
        top10 = question_similarity_top10[q_ID]

        # score these ICL qIDs based on similarity in distribution to q_ID
        # first separate the qIDs with the equivalent num MC and not equivalent num MC 
        easy, easy_hard = [], []
        n_qID_MC = len(icl_data[q_ID]['MC_options'])
        for icl_qID in top10: 
            if n_qID_MC == len(icl_data[icl_qID]['MC_options']): easy.append(icl_qID)
            else: easy_hard.append(icl_qID)
        
        if len(easy_hard)==5:
            return easy_hard

        # if len(easy)<len(easy_hard): There are too many that have different MC options
        # Since we want to keep the more challenging ones in easy_hard, move the textually similar ones to easy 
        # the IDs in easy_hard are ranked from most textually similar to least 
        if len(easy)<len(easy_hard):
            num_to_transfer = len(easy_hard) - 5
            easy = easy + easy_hard[:num_to_transfer]
            easy_hard = easy_hard[num_to_transfer:]

        # if len(easy)>len(easy_hard): split based on similarity to ground truth distribution 
        if len(easy)>len(easy_hard):
            num_to_transfer = len(easy) - 5
            distrib_dist = []
            q_ID_values = np.array(list(icl_data[q_ID][demographic].values()))/np.sum(list(icl_data[q_ID][demographic].values()))
            for easy_qID in easy: 
                icl_values = []
                # calculate distributional difference
                for MC_option in icl_data[easy_qID]['MC_options']:
                    if MC_option in icl_data[easy_qID][demographic].keys(): 
                        icl_values.append(icl_data[easy_qID][demographic][MC_option])
                    else: icl_values.append(0)
                    
                icl_values = np.array(icl_values)/np.sum(icl_values)
                try: distrib_dist.append(total_variation(icl_values, q_ID_values))
                except: distrib_dist.append(1)
            sorted_pairs = sorted(zip(distrib_dist, easy))
            # order qIDs based on smallest to largest distribution difference
            sorted_list_qIDs= [pair[1] for pair in sorted_pairs]

            # keep the ones in easy that have the lowest distributional differences 
            easy = sorted_list_qIDs[:5]
            easy_hard = easy_hard + sorted_list_qIDs[5:]

    elif dataset=='global_values': 

        data_path = '{}/globalvalues/question_similarity.json'.format(os.getcwd())
        f = open(data_path)

        question_similarity_top10 = json.load(f)
        top5 = list(question_similarity_top10[q_ID][demographic].keys())[:5]
        easy_hard = top5

    elif dataset=='nytimes': 
        f = open(data_path + '/question_similarity_top10.json')
        question_similarity_top10 = json.load(f)
        top10 = question_similarity_top10[q_ID]

        # score these ICL qIDs based on similarity in distribution to q_ID
        # first separate the qIDs with the equivalent num MC and not equivalent num MC 
        easy, easy_hard = [], []
        n_qID_MC = len(icl_data[q_ID]['MC_options'])
        for icl_qID in top10: 
            if n_qID_MC == len(icl_data[icl_qID]['MC_options']): easy.append(icl_qID)
            else: easy_hard.append(icl_qID)
        
        if len(easy_hard)==5:
            return easy_hard

        # if len(easy)<len(easy_hard): There are too many that have different MC options
        # Since we want to keep the more challenging ones in easy_hard, move the textually similar ones to easy 
        # the IDs in easy_hard are ranked from most textually similar to least 
        if len(easy)<len(easy_hard):
            num_to_transfer = len(easy_hard) - 5
            easy = easy + easy_hard[:num_to_transfer]
            easy_hard = easy_hard[num_to_transfer:]

        # if len(easy)>len(easy_hard): split based on similarity to ground truth distribution 
        if len(easy)>len(easy_hard):
            num_to_transfer = len(easy) - 5
            q_ID_values = np.array(list(icl_data[q_ID][demographic].values()))/np.sum(list(icl_data[q_ID][demographic].values()))
            for easy_qID in easy: 
                icl_values = []
                # calculate distributional difference
                for MC_option in icl_data[easy_qID]['MC_options']:
                    if MC_option in icl_data[easy_qID][demographic].keys(): 
                        icl_values.append(icl_data[easy_qID][demographic][MC_option])
                    else: icl_values.append(0)
                    
                icl_values = np.array(icl_values)/np.sum(icl_values)

            sorted_pairs = sorted(zip(distrib_dist, easy))
            # order qIDs based on smallest to largest distribution difference
            sorted_list_qIDs= [pair[1] for pair in sorted_pairs]

            # keep the ones in easy that have the lowest distributional differences 
            easy = sorted_list_qIDs[:5]
            easy_hard = easy_hard + sorted_list_qIDs[5:]
    
    else: breakpoint()

    if len(easy_hard)==5:
            return easy_hard

    return 

def get_prompt_header(
    demographic_in_prompt, output_type
):
    prompt = ''
    if output_type=='sequence':
        prompt+= 'Please simulate 30 samples from a group of {} for the question asked. Please only respond with 30 multiple choice answers, no numbering, no new line, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted.'.format(demographic_in_prompt)    
    elif output_type=='model_logprobs': 
        prompt += 'Please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no numbering, no new line, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted.'.format(demographic_in_prompt)
    elif output_type=='express_distribution': 
        prompt += 'Please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no numbering, no new line, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted.'.format(demographic_in_prompt)

    prompt+="\n\nGiven the `question`, produce the fields `answer`.\n\n------\n\n"
    return prompt

def get_prompt_header_few_shot(
    demographic_in_prompt, output_type
):
    prompt = "In this task you will receive information on the distribution of responses from a group of {}s to related survey questions. Given this data, your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)
    prompt+= "First, I will provide the distribution of responses from a group of {}s to a series of questions in a section titled 'Data'. Afterwards, I will provide 5 example responses to the question to help you understand the formatting of this task. ".format(demographic_in_prompt)

    if output_type=='sequence':
        prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
    elif output_type=='model_logprobs': 
        prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
    elif output_type=='express_distribution': 
        prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)

    return prompt    

# Get all in context learning prompts for dataset opinionqa
def get_zeroshot_prompt_opinionqa(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_500", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    output_type="model_logprobs"
):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    demographic_in_prompt = demographic
    data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    prompt = get_prompt_header(demographic_in_prompt, output_type)

    prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
    for i, option in enumerate(list(data[q_ID][demographic].keys())):
        prompt +="{}. {}. ".format(options[i], option)

    return prompt

def get_icl_prompt_opinionqa(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_100", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    output_type="model_logprobs"
):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    demographic_in_prompt = demographic
    data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    prompt = get_prompt_header_few_shot(demographic_in_prompt, output_type)
    prompt+="\n\nGiven the fields 'context` and `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"

    # we need the larger set to get icl demos
    if wave == 'Pew_American_Trends_Panel_disagreement_100':
        icl_wave='Pew_American_Trends_Panel_disagreement_500'
    icl_data = json.load(open(data_path + icl_wave + '/' + demographic_group + "_data.json"))

    # get icl qids
    ICL_qIDS = get_ICL_qIDs(
        q_ID=q_ID, wave=icl_wave, 
        demographic_group=demographic_group, demographic=demographic)

    for icl_qID in ICL_qIDS:
        if icl_qID == q_ID:
            continue
        n = (sum(icl_data[icl_qID][demographic].values()))
        MC_options = list(icl_data[icl_qID][demographic].keys())
        all_options, probs = [], []
        for i, option in enumerate(MC_options):
            all_options.append(options[i])
            probs.append(icl_data[icl_qID][demographic][option]/n)

        prompt+= "\nQuestion: " + icl_data[icl_qID]['question_text'] + "?\n"
        for i, option in enumerate(MC_options):
            prompt +="{}. {}. ".format(options[i], option)

        if output_type=='sequence':
            # Generate 30 flips
            try: 
                flips = random.choices(all_options, probs, k=30)
                prompt+="\nAnswer: " + ' '.join(flips) + '\n\n------\n\n'
            except: prompt+=''
            
        elif output_type=='model_logprobs': 
            try: 
                flips = random.choices(all_options, probs, k=1)
                prompt+="\nAnswer: " + ' '.join(flips) + '\n\n------\n\n'
            except: prompt+=''
            
        elif output_type=='express_distribution': 
            prompt +="\nAnswer: {"
            for i, prob in enumerate(probs):
                prompt+="'{}': '{}%', ".format(all_options[i], int(prob*100))
            prompt = prompt[:-2] + '}\n\n------\n\n' # -2 to get rid of last space

    prompt+='Your turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)

    prompt+= "Question: " + icl_data[q_ID]['question_text'] + "?\n"
    for i, option in enumerate(list(icl_data[q_ID][demographic].keys())):
        prompt +="{}. {}. ".format(options[i], option)
    prompt+="\nAnswer:"

    return prompt

# Get all in context learning prompts for dataset nytimes
def get_icl_prompt_nytimes(q_ID, wave, demographic_group, demographic, output_type):
    prompt_names = {"Republican" : 'a Republican person', "Democrat" : 'a Democrat person', "Male" : 'a man', "Female" : 'a woman'}
    if demographic in prompt_names.keys(): demographic_in_prompt = prompt_names[demographic]
    demographic_in_prompt = demographic
    prompt = get_prompt_header_few_shot(demographic_in_prompt, output_type)
    prompt+="\n\nGiven the fields `context`, `Book Title`, `Book Genre`, `Book Summary`, `question`, produce the fields `answer`.\n\n------\n\n"

    # ICL Is all other question in the wave 
    
    data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    icl_data = data
    data_path = '{}/nytimes/'.format(os.getcwd())
    ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, wave, demographic, data_path, dataset='nytimes')
    for icl_qID in ICL_qIDS:
        if icl_qID != q_ID:
            
            all_options, probs = [], []
            prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)

            MC_options = data[icl_qID]['MC_options']
            n = (sum(data[icl_qID][demographic].values()))
            
            for i, option in enumerate(MC_options):
                if str(option) in data[icl_qID][demographic]: 
                    all_options.append(options[i])
                    probs.append(data[icl_qID][demographic][option]/n)
                    prompt +="'{}' be {}%, ".format(option, int((data[icl_qID][demographic][option]/n)*100))
            prompt+= ".\nBook Title: " + icl_qID 
            prompt+= "\nBook Genre: " + data[icl_qID]['genre'] 
            prompt+= "\nBook Summary: " + data[icl_qID]['summary'] 
            prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)
            for i, option in enumerate(MC_options):
                prompt +="'{}'. {}\n".format(options[i], option)
            if output_type=='sequence':
                # Generate 30 flips
                try: 
                    flips = random.choices(all_options, probs, k=30)
                    prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                except: prompt+=''
                
            elif output_type=='model_logprobs': 
                try: 
                    flips = random.choices(all_options, probs, k=1)
                    prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                except: prompt+=''
            elif output_type=='express_distribution': 
                prompt +="Answer: {"
                for i, prob in enumerate(probs):
                    prompt+="{}: '{}%', ".format(all_options[i], int(prob*100))
                prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

    prompt+='\nYour turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)

    prompt+= "\nBook Title: " + q_ID 
    prompt+= "\nBook Genre: " + data[q_ID]['genre'] 
    prompt+= "\nBook Summary: " + data[q_ID]['summary'] 
    prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)

    
    for i, option in enumerate(list(data[q_ID][demographic].keys())):
        prompt +="'{}'. {}\n".format(options[i], option)
    prompt+="Answer:"
            
    return prompt

def get_icl_prompt_global_values(q_ID, wave, demographic_group, demographic, output_type):
    demographic_in_prompt = demographic
    prompt = get_prompt_header_few_shot(demographic_in_prompt, output_type)
    prompt+="\n\nGiven the fields 'context` and `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"

    # ICL Is all other question in the wave 
    data_path = '{}/global_values/'.format(os.getcwd())
    f = open('{}/{}/{}_data.json'.format(data_path, wave, demographic_group))
    data = json.load(f)

    ICL_qIDS = get_ICL_qIDs(q_ID, wave, demographic, data_path)
    qID_to_wave = json.load(open('{}/qID_to_wave.json'.format(data_path)))
    for icl_qID in ICL_qIDS:
        if icl_qID != q_ID:
            df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
            ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
            ICL_qID_wave = ICL_qID_wave.iloc[0][4:]

            f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
            q_ID_data = json.load(f) # this is specific to the new wave 
            all_options, probs = [], []
            prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)
            n = (sum(q_ID_data[icl_qID][demographic].values()))
            MC_options = list(q_ID_data[icl_qID][demographic].keys())
            for i, option in enumerate(MC_options):
                all_options.append(options[i])
                probs.append(q_ID_data[icl_qID][demographic][option]/n)
                prompt +="'{}' be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
            prompt+= "\nQuestion: " + q_ID_data[icl_qID]['question_text'] + "?\n"
            for i, option in enumerate(MC_options):
                prompt +="'{}'. {}\n".format(options[i], option)
            if output_type=='sequence':
                # Generate 30 flips
                try: 
                    flips = random.choices(all_options, probs, k=30)
                    prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                except: prompt+=''
                
            elif output_type=='model_logprobs': 
                try: 
                    flips = random.choices(all_options, probs, k=1)
                    prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                except: prompt+=''
            elif output_type=='express_distribution': 
                prompt +="Answer: {"
                for i, prob in enumerate(probs):
                    prompt+="'{}': '{}%', ".format(all_options[i], int(prob*100))
                prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

    prompt+='\nYour turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)

    prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
    for i, option in enumerate(list(data[q_ID][demographic].keys())):
        prompt +="'{}'. {}\n".format(options[i], option)
    prompt+="Answer: "    
        
    return prompt

def get_few_shot_training_examples(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_100", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    output_type="model_logprobs",
    dataset="opinionqa",
    n_shots=5,
    n_simulations_per_shot=1,
    provide_ground_truth_distribution=False
):
    
    if dataset=='opinionqa': 
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
        # we need the larger set to get icl demos
        if wave == 'Pew_American_Trends_Panel_disagreement_100':
            icl_wave='Pew_American_Trends_Panel_disagreement_500'
        icl_data = json.load(open(data_path + icl_wave + '/' + demographic_group + "_data.json"))
    elif dataset=='nytimes':
        data_path = '{}/{}'.format(os.getcwd(), dataset)
        data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
        icl_data = data
    else: 
        data_path = '{}/{}'.format(os.getcwd(), dataset)
        data = json.load(open(data_path + wave + '/' + demographic_group + "data.json"))
        icl_data = data

    demographic_in_prompt = demographic
    prompt = get_prompt_header(demographic_in_prompt, output_type)

    # get icl qids
    ICL_qIDS = get_ICL_qIDs(
        q_ID=q_ID, wave=icl_wave, 
        demographic_group=demographic_group, demographic=demographic)

    examples = []
    
    for icl_qID in ICL_qIDS[:n_shots]:
        if icl_qID == q_ID:
            continue

        n = (sum(icl_data[icl_qID][demographic].values()))
        MC_options = list(icl_data[icl_qID][demographic].keys())
        all_options, probs = [], []
        for i, option in enumerate(MC_options):
            all_options.append(options[i])
            probs.append(icl_data[icl_qID][demographic][option]/n)
            if provide_ground_truth_distribution:
                prompt +="{} be {}%, ".format(option, int((icl_data[icl_qID][demographic][option]/n)*100))

        example_input = prompt + "\nQuestion: " + icl_data[icl_qID]['question_text'] + "?\n"
        for i, option in enumerate(MC_options):
            example_input +="{}. {}. ".format(options[i], option)

        for _ in range(n_simulations_per_shot):
            example_output = ""
            if output_type=='sequence':
                # Generate 30 flips
                try: 
                    flips = random.choices(all_options, probs, k=30)
                    example_output+="Answer: " + ' '.join(flips)
                except: example_output+=''
                
            elif output_type=='model_logprobs': 
                try: 
                    flips = random.choices(all_options, probs, k=1)
                    example_output+="Answer: " + ' '.join(flips)
                except: example_output+=''
                
            elif output_type=='express_distribution': 
                example_output +="Answer: {"
                for i, prob in enumerate(probs):
                    example_output+="'{}': '{}%', ".format(all_options[i], int(prob*100))
                example_output = example_output[:-2] + '}' # -2 to get rid of last space
            examples.append([example_input, example_output, q_ID, icl_qID, demographic_group, demographic, output_type, wave])
    return pd.DataFrame(examples, columns=[
        'input', 'output', 'qID', 'icl_qID', 'demographic_group', 'demographic', 'output_type', 'wave'
    ])

def get_test_questions_with_distributions(
    seen_qIDs,
    wave="Pew_American_Trends_Panel_disagreement_500", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    data_path='{}/opinions_qa/data/human_resp/'.format(os.getcwd()),
    dataset="opinionqa"
):
    demographic_in_prompt = demographic
    if dataset == "opinionqa" or dataset == "nytimes":
        data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    else:
        data = json.load(open(data_path + '/' + demographic_group + "data.json"))
    filtered_data = {}
    for k, v in data.items():
        if k in seen_qIDs:
            continue
        filtered_data[k] = v
    return filtered_data    

def prepare_df(original_df, tokenizer):
    def apply_chat_template(row):
        messages = [{"role": "user", "content": row["input"]}]
        nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
        return tokenizer.decode(nobos)
    
    original_df['input'] = original_df.apply(apply_chat_template, axis=1)
    return original_df # do nothing, the task will be standard instruction tuning.

def parse_answers(raw_response, available_choices, answer_tag=True, output_type="sequence"):
    """
    Parse the answers from a raw response string and calculate counts and probabilities.

    Args:
        raw_response (str): The raw input string containing the answers. 
        If output_type is "sequence", raw_response is a string. e.g. "A B C D E F"
        If output_type is "express_distribution", raw_response is a dictionary. e.g. {"A": '12%', "B": '23%', "C": '34%'}
        available_choices (list): A list of valid answer choices (e.g., ["A", "B", "C", "D", "E", "F"]).

    Returns:
        tuple: (status, result)
            - status (bool): True if successful, False if an error occurs.
            - result: A dictionary containing counts and probabilities if successful,
                      or an error message if an error occurs.
    """
    try:
        if answer_tag and "Answer:" not in raw_response:
            raise ValueError("No 'Answer:' keyword found in input.")
        if answer_tag:
            answers_part = raw_response.split("Answer:")[1]
        else:
            answers_part = raw_response.strip()

        if output_type == "sequence":
            answers_list = answers_part.strip().split()
            if not answers_list:
                raise ValueError("No parsable answers found in input.")
            counts = {choice: 0 for choice in available_choices}
            total_answers = 0

            for answer in answers_list:
                answer = answer.strip()
                if answer in available_choices:
                    counts[answer] += 1
                    total_answers += 1
                else:
                    # Skip invalid choices
                    pass
            if total_answers < 3:
                raise ZeroDivisionError("Not enough valid answers to calculate probabilities.")
            probabilities = {choice: count / total_answers for choice, count in counts.items()}
            return True, {"counts": counts, "probabilities": probabilities}
        elif output_type == "express_distribution":
            probabilities = {key: float(value.strip('%')) / 100 for key, value in raw_response.items()}
            return True, {"probabilities": probabilities}
            
    except ValueError as ve:
        return False, {"message": str(ve)}
    except ZeroDivisionError as zde:
        return False, {"message": str(zde)}
    except Exception as e:
        return False, {"message": f"Unexpected error: {str(e)}"}

def calculate_l1(golden_distribution, predicted_distribution):
    golden_probs = np.array([golden_distribution[key] for key in golden_distribution])
    predicted_probs = np.array([predicted_distribution[key] for key in golden_distribution])
    return np.sum(np.abs(golden_probs - predicted_probs))

def compute_l1_values(dist_pairs):
    return [calculate_l1(dist_pair[0], dist_pair[1]) for dist_pair in dist_pairs]