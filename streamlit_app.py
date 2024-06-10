import streamlit as st
import numpy as np
import json
import time
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Code Efficiency
import cProfile
import pstats

# imports
from audio_processing import *
from visualization import *
from utils import *
import json
import pandas as pd
#from dotenv import load_dotenv, find_dotenv
import os
import logging
#from utils import load_NER_template, load_sentiment_template, load_file, runNER, runSentiment
from dotenv import load_dotenv, find_dotenv
import openai
#from llama import LLM

# Lanchain
from typing import Optional
import openai
import os
import langchain
from dotenv import load_dotenv, find_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import  convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.adapters.openai import ChatCompletion
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# Streamlit
import streamlit as st
import tempfile
import os

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


class Transcripts(BaseModel):
    """A contraceptive call center transcripts ."""
    speaker: str = Field(description="Capture speakers (Agent and Client) in the transcript. Output either as Agent or Client")
    #location: str = Field(description="Capture the location the client is calling from")
    # brand: str = Field(description="names of brands mentioned in the transcript")
    product: str = Field(description="names of family planning products mentioned in the transcript")
    product_side_effect: str = Field(description="From the transcripts, extract the side effects mentioned  ")
    Emotions: str = Field(description="""Using the following emotions [ Fulfilled, Enthusiastic, Anxious, Neutral, Unfulfilled, scared], output the emotion expressed by the client at the beginning of the conversation and the emotion at the end of the conversation. Use the format.

                    Emotion at the beginning of conversation: 
                    Emotion at the end at the end of conversation:""")
    Entities_mentioned: str = Field(description="""Extract the following information from the call center call text, If information is not mentioned, write "Not Mentioned", do not add any extra information. Return in JSON format like:
            {{
            "State": State where the call was made,
            "Product": Family planning products mentioned,
            "Caller's Occupation": Caller's occupation,
            "Method": Family planning method mentioned,
            "Method Duration": Family planning method duration mentioned,
            "Side Effects": Family planning side effects mentioned
            }}""")
    # time: Optional[int] = Field(description="timestamps of each speaker from the transcription")
    #sentiment: str = Field(description="Extract sentiments (positive, negative or neutral) for each speaker in the transcription.")
    sentiment: str = Field(description="""Extract the caller's sentiment [happy, neutral, uncertain, anxious, scared] at the beginning and end of the call based on their words and sentences.State reason for your response. if you are unsure, write "Uncertain". All output should be in JSON format like:

        {{"Beginning Sentiment": The classification,
         "Reason": The reason why you picked that,
        "Ending Sentiment": The classification,
         "Reason": The reason why you picked that
        }}""")
    compliance: str = Field(description="""Extract the following compliance report from the call center text. Write “yes” if the If information is mentioned, “no” if not mentioned, do not add any extra information. Return in JSON.

        {{“Establish Cordial Relationship”: if the call agent welcomes the client,
        “Client Preference”: If the agent ask for the client preferred method/product,
        “Client Satisfaction”: If the agent request feedback information from the client to ensure the correct information is received
        }}""")
    gender: str = Field(description="Extract the genders of the two speakers")
    # client_marital_status: str = Field(description="Is the client single or married")
    # pregnancy_status: str = Field(description="Is the client pregnant or normal")
    #age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
    """Information to extract."""
    people: List[Transcripts] = Field(description="List of info about transcripts")


def main():
    st.title('Speaker Diarization | Named Entity and Sentiment Extraction System')
    
    uploaded_audio = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'mp4'])
    
    if uploaded_audio is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_audio.name.split('.')[-1]) as temp_file:

            temp_file.write(uploaded_audio.getvalue())
            temp_file_path = temp_file.name
    
        wav_path = temp_file_path
        # st.write(wav_path)
            
        if not temp_file_path.lower().endswith('.wav'):
            logging.info("Converting audio to WAV format...")
            wav_path = convert_to_wav(temp_file_path)
            os.unlink(temp_file_path)

        st.audio(wav_path, format='audio/wav')

        logging.info("Transcribing audio...")
        with st.spinner('Transcribing the audio...'):
            segments = transcribe_audio(wav_path, model_size='large')
            st.success('Audio Transcription completed and output saved.')
    
        y, sr = load_audio(wav_path)
        duration, frames, rate = get_audio_metadata(y, sr)
        st.write(f"Frames: {frames}")
        st.write(f"Frame Rate: {rate} frames/sec")
        st.write(f"Duration: {duration} seconds")


        html_transcript_filename = "transcript.html"
        transcript_filename = "transcript.txt"
 
        with open(html_transcript_filename, "w", encoding='utf-8') as f:
            f.write("<html><body>")
            for segment in segments:
                f.write(f"<p>{segment['text']}</p>")
            f.write("</body></html>")
 
        with open(transcript_filename, "w", encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['text']}\n")
 
        with open(html_transcript_filename, "r", encoding='utf-8') as f:
            transcript_html = f.read()
            st.markdown(transcript_html, unsafe_allow_html=True)
 
        logging.info("Transcript saved successfully.")
        time.sleep(10)

        # embeddings = np.zeros(shape=(len(segments), 192))
        # for i, segment in enumerate(segments):
        #     embeddings[i] = segment_embedding(wav_path, duration, segment)
        # embeddings = np.nan_to_num(embeddings)
        # labels = cluster_speakers(embeddings, 2)

        # for i, segment in enumerate(segments):
        #     segment["speaker"] = labels[i]

        # speaker_roles = identify_roles(segments)

        # html_transcript_filename = "transcript.html"
        # transcript_filename = "transcript.txt"

        # save_html_transcript(segments, speaker_roles, html_transcript_filename)
        # save_transcript(segments, speaker_roles, transcript_filename)

        # with open(html_transcript_filename, "r", encoding='utf-8') as f:
        #     transcript_html = f.read()
        #     st.markdown(transcript_html, unsafe_allow_html=True)
            


        # Assign speaker labels to segments
        # for i in range(len(segments)):
        #     segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Save transcript
        # save_transcript(segments)

        # transcript_filename = "transcript3.txt"
        # with open(transcript_filename, "r", encoding='utf-8') as f:
        #     transcript = f.read()
        #     st.text_area("Transcript", transcript, height=300)
        
        # logging.info("Transcript saved successfully.")
        # time.sleep(10)

        # # Extraction model

        with st.container():
            st.header("Data Extraction Results")
            with st.spinner('Extracting information from the transcript...'):
                model = ChatOpenAI(temperature=0)
                extraction_functions = [convert_pydantic_to_openai_function(Information)]
                extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Think carefully, and then tag the text as instructed, don't assume anything if you are not sure"),
                    ("user", "{input}")
                ])
                extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

                output = extraction_chain.invoke({"input": ' '.join(s['text'] for s in segments)})

                df = pd.json_normalize(output['people'])
                st.dataframe(df)


                # st.json(output)

                with open('output3.json', 'w') as outfile:
                    json.dump(output, outfile, ensure_ascii=False, indent=4)
                st.success('Data extraction complete and output saved.')

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    #main()