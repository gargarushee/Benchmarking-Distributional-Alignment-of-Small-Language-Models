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
def get_q_IDs(wave="Pew_American_Trends_Panel_disagreement_100"):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    path = open(data_path + wave + '/info.csv')
    df = pd.read_csv(path)
    waves = np.array(df['survey'])
    q_IDs = np.array(df['key'])
    return q_IDs, waves

    
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

def get_icl_prompt(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_100", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    output_type="model_logprobs"
):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    demographic_in_prompt = demographic
    data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    prompt = "In this task you will receive information on the distribution of responses from a group of {}s to related survey questions. Given this data, your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)
    prompt+= "First, I will provide the distribution of responses from a group of {}s to a series of questions in a section titled 'Data'. Afterwards, I will provide 5 example responses to the question to help you understand the formatting of this task. ".format(demographic_in_prompt)

    if output_type=='sequence':
        prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
    elif output_type=='model_logprobs': 
        prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
    elif output_type=='express_distribution': 
        prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)

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
            prompt +="{} be {}%, ".format(option, int((icl_data[icl_qID][demographic][option]/n)*100))

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


def get_few_shot_training_examples(
    q_ID, 
    wave="Pew_American_Trends_Panel_disagreement_100", 
    demographic_group="POLPARTY",
    demographic="Democrat",
    output_type="model_logprobs",
    n_shots=5,
    n_simulations_per_shot=1,
    provide_ground_truth_distribution=False
):
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    demographic_in_prompt = demographic
    data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
    prompt = "Your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)

    if output_type=='sequence':
        prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted.'.format(demographic_in_prompt)
    elif output_type=='model_logprobs': 
        prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted.'.format(demographic_in_prompt)
    elif output_type=='express_distribution': 
        prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted.'.format(demographic_in_prompt)

    # we need the larger set to get icl demos
    if wave == 'Pew_American_Trends_Panel_disagreement_100':
        icl_wave='Pew_American_Trends_Panel_disagreement_500'
    icl_data = json.load(open(data_path + icl_wave + '/' + demographic_group + "_data.json"))

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