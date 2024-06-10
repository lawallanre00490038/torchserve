import json
import re
#from llama_cpp import Llama
#import langchain
#from langchain.llms import 

import datetime

def format_timestamp(timestamp):
    return str(datetime.timedelta(seconds=timestamp))

def identify_roles(segments):
    potential_agent = None
    potential_client = None

    for segment in segments[:5]:  # Analyze the first 5 segments
        text = segment["text"].lower()
        
        # Check for agent-specific phrases
        if "how may i help you" in text or "thank you for calling" in text:
            potential_agent = segment["speaker"]
        
        # Check for client-specific phrases
        if "i have a problem with" in text or "i'm calling about" in text:
            potential_client = segment["speaker"]

        # Break if both roles are identified
        if potential_agent is not None and potential_client is not None:
            break

    # Handle cases where only one role is identified
    if potential_agent is not None and potential_client is None:
        # The other speaker must be the client
        potential_client = 1 - potential_agent
    elif potential_client is not None and potential_agent is None:
        # The other speaker must be the agent
        potential_agent = 1 - potential_client
    elif potential_agent is None and potential_client is None:
        # Default assumption if neither role is clear
        potential_agent = segments[0]["speaker"]
        potential_client = 1 - potential_agent

    return {potential_agent: "Agent", potential_client: "Client"}


def save_transcript(segments, speaker_roles, filename="transcript3.txt"):
    with open(filename, "w", encoding='utf-8') as f:
        for segment in segments:
            speaker_label = "Agent" if speaker_roles[segment["speaker"]] == "Agent" else "Client"
            f.write("\n" + speaker_label + ' ' + format_timestamp(segment["start"]) + '\n')
            f.write(segment["text"][1:] + ' ')


def save_html_transcript(segments, speaker_roles, filename="transcript.html"):
    with open(filename, "w", encoding='utf-8') as f:
        for segment in segments:
            speaker_label = speaker_roles[segment["speaker"]]
            color = "blue" if speaker_label == "Agent" else "green"
            timestamp = format_timestamp(segment["start"])
            f.write(f'<div style="color: {color};"><strong>{speaker_label} {timestamp}</strong>: {segment["text"]}</div>\n')

# def save_transcript(segments, filename="transcript3.txt"):
#     with open(filename, "w", encoding='utf-8') as f:
#         for (i, segment) in enumerate(segments):
#             if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
#                 f.write("\n" + segment["speaker"] + ' ' + format_timestamp(segment["start"]) + '\n')
#             f.write(segment["text"][1:] + ' ')






# def process_call_transcription(file_path):
#     # Initialize the Llama model (assuming Llama class and initialization as per your environment/setup)
#     llm = Llama(model_path="models/llama-2-7b.ggmlv3.q5_K_S.bin.gguf", 
#                 n_ctx=8192, 
#                 n_batch=512)

#     # Read the call transcription text from the file
#     with open(file_path, 'r') as file:
#         call_text = file.read()

#     print("Call Text:")
#     print(call_text)
#     print("--------------------------------------------")

#     # Templates are not filled since we are manually parsing the raw text
#     # Define patterns to match the required information
#     state_pattern = r'State where the call was made\s+-\s+"(.+?)"'
#     product_pattern = r'Family planning products mentioned\s+-\s+"(.+?)"'
#     qa_pair_pattern = r'Questions asked, reply given pairs\s+-\s+"(.+?)"\s+-\s+"(.+?)"'

#     # Use regular expressions to find matches in the call text
#     state_match = re.search(state_pattern, call_text)
#     product_match = re.search(product_pattern, call_text)
#     qa_pair_matches = re.findall(qa_pair_pattern, call_text)

#     # Extract the information
#     state = state_match.group(1) if state_match else None
#     product = product_match.group(1) if product_match else None
#     questions_and_replies = [list(match) for match in qa_pair_matches] if qa_pair_matches else []

#     # Construct the JSON object for NER
#     ner_result = {
#         "State": state,
#         "Product": product,
#         "Questions and Replies": questions_and_replies
#     }

#     # Here you would include the sentiment analysis code which I have not included
#     # assuming it follows a similar structure for parsing raw text.

#     # Assuming sentiment_result is obtained in a similar way as ner_result
#     sentiment_result = {}  # Replace this with actual sentiment analysis logic

#     # Combine results
#     combined_results = {
#         'EntityExtraction': ner_result,
#         'SentimentAnalysis': sentiment_result
#     }

#     return combined_results


# import json
# import time
# import os
# import logging
# from llama_cpp import Llama


# llm = Llama(model_path="models/llama-2-13b.ggmlv3.q4_0.bin.gguf",
#                 n_ctx=8192,
#                 n_batch=512)

# def process_call_transcription(file_path):
#     # Initialize the Llama model
#     llm = Llama(model_path="models/llama-2-7b.ggmlv3.q4_K_S.bin.gguf",
#                 n_ctx=8192,
#                 n_batch=512)

#     # Wait for the file to be fully written if not yet available
#     while not os.path.exists(file_path):
#         logging.info(f"Waiting for the file {file_path} to exist.")
#         time.sleep(2)  # checks every 2 seconds

#     # Ensure the file is not being written to by checking its size consistency
#     size_before = -1
#     while size_before != os.path.getsize(file_path):
#         size_before = os.path.getsize(file_path)
#         time.sleep(1)  # wait 1 second
#         if size_before == os.path.getsize(file_path):
#             break  # if the size hasn't changed, the file is ready to read

#     # Read the call transcription text from the file
#     with open(file_path, 'r') as file:
#         call_text = file.read()

#     print("Call Text:")
#     print(call_text)
#     print("--------------------------------------------")

#     # Define the templates for entity extraction and sentiment analysis
#     ner_template = """
#     Extract the following information from a call center call text return in JSON format:
#     1. State where the call was made
#     2. Family planning products mentioned
#     3. Questions asked, reply given pairs
#     """

#     sentiment_template = """
#     Classify whether the caller in this call is [satisfied, unsatisfied] with the response given by the call center agent. Also state the reason for this. If you are unsure, write Uncertain. All output should be in JSON format.

#     """

#     # Extract entities
#     ner_prompt = ner_template.format(call_text=call_text)
#     ner_output = llm(ner_prompt, max_tokens=2048, echo=False, temperature=0.2, top_p=0.1)

#     with open('raw_ner_output.txt', 'w') as file:
#         file.write(ner_output['choices'][0]['text'])

#     try:
#         ner_result = json.loads(ner_output['choices'][0]['text'])
#     except json.JSONDecodeError as e:
#         # If there's an error, print the message and the problematic part of the string
#         print("JSONDecodeError:", e)

#     # Analyze sentiment
#     sentiment_prompt = sentiment_template.format(call_text=call_text)
#     sentiment_output = llm(sentiment_prompt, max_tokens=2048, echo=False, temperature=0.2, top_p=0.1)
#     # sentiment_result = json.loads(sentiment_output['choices'][0]['text'])

#     with open('raw_sentiment_output.txt', 'w') as file:
#         file.write(ner_output['choices'][0]['text'])

#     try:
#         sentiment_result = json.loads(sentiment_output['choices'][0]['text'])
#     except json.JSONDecodeError as e:
#         # If there's an error, print the message and the problematic part of the string
#         print("JSONDecodeError:", e)

#     # Combine results
#     combined_results = {
#         'EntityExtraction': ner_result,
#         'SentimentAnalysis': sentiment_result
#     }

#     return combined_results


# def load_llama():
#     # Initialize the Llama model
#     llm = Llama(model_path="models/llama-2-7b.ggmlv3.q5_K_S.bin.gguf",
#                 n_ctx=8192,
#                 n_batch=512)
    
#     return llm


# import json

# def convertOutputtoDict(text, start="{", stop="}"):
#     """
#     Converts text input to a dictionary by extracting the JSON portion.

#     Args:
#         text (str): The input text that contains JSON.
#         start (str): The character where the JSON starts (default is "{").
#         stop (str): The character where the JSON stops (default is "}").

#     Returns:
#         dict: The JSON portion of the text converted to a dictionary.
#     """
#     # Get the location of the JSON output
#     start_index = text.index(start)
#     stop_index = text.index(stop, start_index)

#     # Extract the JSON string
#     json_str = text[start_index:stop_index+1]

#     # Convert the JSON string to a dictionary
#     data = json.loads(json_str)

#     return data



# def applytemplate(prompt_template, text):
#     '''
#     Applies a prompt template to a text

#     Args:
#     ----
#     prompt_template: The prompt template for the task
#     text: The text to be applied

#     Returns:
#     -------
#     The prompt with the actual call transcript
#     '''
#     return prompt_template.replace("<<<CALLINFO>>>", text)


# def runNER(text, ner_template):
#     # Perform NER using the updated NER template
#     prompt = applytemplate(ner_template, text)
#     # Get the response from LLM model
#     #response = llm.chat(prompt)
#     response = llm(prompt, max_tokens=2048, echo=False, temperature=0.2, top_p=0.1)
#     # Convert output to Dictionary
#     return response # convertOutputtoDict(response)

# def runSentiment(text, sentiment_template):
#     # Perform Sentiment Analysis using the updated Sentiment template
#     prompt = applytemplate(sentiment_template, text)
#     # Get the response from LLM model
#     # response = llm.chat(prompt)
#     response = llm(prompt, max_tokens=2048, echo=False, temperature=0.2, top_p=0.1)
#     # Convert output to Dictionary
#     return response

# def load_NER_template(filepath="prompt_template/NERtemplate.txt"):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         ner_template = file.read()
#     return ner_template

# def load_sentiment_template(filepath="prompt_template/Sentiment.txt"):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         sentiment_template = file.read()
#     return sentiment_template

# def load_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         return file.read()


# Usage example
# file_path = 'transcript.txt'
# results = process_call_transcription(file_path)
# print(json.dumps(results, indent=4))