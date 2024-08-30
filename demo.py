# @title Create your app
# @markdown # Retrieve reference files
# %%writefile app.py

import os
import sys
import datetime
from dotenv import load_dotenv
import streamlit as st
import tempfile
import soundfile as sf
import google.generativeai as genai
from API_KEY import API_KEY


from audio_recorder_streamlit import audio_recorder

# Vertex AI imports
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Content,
)

# Google Cloud Storage
from google.cloud import storage

# Environment variables
# @markdown ### Set your environment
load_dotenv()
#PROJECT_ID = "mythical-lens-406709"  # @param {type:"string"}
#LOCATION = "asia-southeast1"  # @param {type:"string"}
# PROJECT_ID = os.environ.get("GCP_PROJECT")
# LOCATION = os.environ.get("GCP_REGION")

# # Initialize Vertex AI model
# vertexai.init(project=PROJECT_ID, location=LOCATION)

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('models/gemini-1.5-pro-001')
# @markdown ### Choose your AI Model
# Configure Vertex AI
# MODEL_ID = "gemini-1.5-pro-001"  # @param ["gemini-1.5-pro-001", "gemini-1.5-flash-001"] {isTemplate:true}
# model = GenerativeModel(MODEL_ID)

# --- Helper Functions ---
def generate_ai_response(contents):
    generation_config = {
                            "max_output_tokens": 8192,
                            "temperature": 0.05,
                            "top_p": 0.95,
                        }

    # safety_settings = {
    #                     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #                     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #                     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #                 }

    # safety_settings=safety_settings

    response = model.generate_content(contents, generation_config=generation_config, )
    if response:
        st.write("Response:")
        st.write(response.text)
# --- End of Helper functions ---

# --- App Variables ---
report_string_array = [
            "Geography lesson - Topic 2.3 Sustainable Tourism Development",
]
your_pdf_storage_prefix = "/home/jiaxinyao/demo-app/Samples/pdf/"
your_pdf_audio_prefix = "/home/jiaxinyao/demo-app/Samples/audio/"
pdf_sample_question_array = [
        "Cite where the arguments made in the audio file can be found in the slides. Give the exact file name and page of the slide.",

    ] # Takes in as many as you want

pdf_prompt = f"""
                Your are a teacher's assistant looking to help make citations and references for classrom discussions to textbook materials.
                Please answer the following questions:
                """
audio_geolocation_url = (
        "https://storage.mtls.cloud.google.com/jx-eg-demo/Geography%20lesson%20-%20audio.MP3"
    )

audio_file_uri = "/home/jiaxinyao/demo-app/Samples/audio/geogLesson.MP3"
audio_type = "audio/mp3"

# UPLOAD FILES
audio_file_tab_uploaded = genai.upload_file(path=audio_file_uri, mime_type="audio/mp3",display_name="geogLesson")


audio_sample_output= """
    | Speaker | Text |
    |---|---|
    | **Teacher** | So the inquiry question is: Mauritius, is tourism the way to go? So we have three groups supporting yes, tourism should be done, and we have three groups supporting no, it shouldn't be done. |
    | **Student A** | We are well aware the tourism sector contribute to one quarter of our local economy. |
    | **Student B** | You guys claim that you guys want to invest more money into tourism, correct? Why don't we invest the money into upgrading our sugarcane plantation so that they are more resistant to rising water temperatures which is caused by the increased carbon footprint which is produced by tourist activity? |
    | **Student C** | Sugarcane is affected by natural disasters. We cannot control them. So what do you want the government to do? Did you read the background information properly? Because it's the natural disaster that is affecting sugarcane and not your carbon footprint. Thank you. |
    | **Teacher** | Okay. No, no, no, no more questions. No, no. No back and forth. Okay. So pass down the reflection sheet. Okay, while we discuss for about five minutes, you complete the post-conference reflections. |    Detailed Summary:

    \n
    This transcription captures a lively classroom debate in Singapore. The central question is whether tourism is the right path for Mauritius.

    \n
    Student A, representing the pro-tourism side, highlights tourism's significant contribution to the local economy.
    Student B, opposing excessive reliance on tourism, challenges the pro-tourism group's proposal to invest more in the sector. They suggest investing in sugarcane plantations instead, aiming to enhance their resilience against climate change impacts, which they argue are exacerbated by tourism.
    Student C, also seemingly against over-dependence on tourism, retorts that natural disasters, not carbon footprint from tourism, are the primary threat to sugarcane. They question the opposing group's grasp of the background information.
    The Teacher, mediating the debate, calls for order, preventing further back-and-forth arguments. They then instruct the students to complete a reflection sheet while a separate discussion takes place.
    The debate showcases critical thinking, with students presenting economic and environmental arguments. The use of Singaporean English is evident in the vocabulary and pronunciation.
    """

tab_1_3_text ="""
Response:

## Arguments for and Against Investing in Tourism in Mauritius

### Argument For Investing in Tourism

- **Tourism's Contribution to the Economy**:
  - The audio states, "The tourism sector contributes to one-quarter of our local economy."
  - This argument is found on slide 2, in the infographic titled **"SUSTAINABLE TOURISM"**.
  - The infographic highlights:
    - "Tourism is responsible for 235 million jobs in the world."
    - Tourism is the "Main income source for many developing countries."

### Argument Against Investing in Tourism

- **Environmental Impact**:
  - The opposing team suggests investing in upgrading sugarcane plantations to be more resistant to rising water temperatures caused by the increased carbon footprint of tourist activity.
  - This argument is based on the information provided on slide 3, which states:
    - "The growth of tourism and sugar plantations industries has [an] impact on the biodiversity of the country."
    - Highlights the "ecological destruction" caused by tourism-related activities.

- **Impact of Natural Disasters**:
  - The audio mentions the impact of natural disasters on sugarcane plantations. However, this point is refuted as not being directly related to the carbon footprint of tourism.
                    """
# --- End of App Variables ---

# --- Prompts ---
# @markdown ### AI Context Prompt
# @markdown Here you can edit the prompts given to the AI Model to give it context on its job \
# @markdown \
# @markdown Edit the prompt to get a more accurate transcription for the objectives \
# @markdown \
# @markdown **Sample prompt**
# @markdown ```
# @markdown You are a transcriber specializing in transcribing Singaporean conversations.
# @markdown Be as accurate as possible in the transcription, give me as many rows as possible.
# @markdown The given audio is in a classroom setting, with a teacher and many students.
# @markdown Please label the teacher as Teacher, and the students as Student A Student B etc etc, and present it in a table form
# @markdown with the fields speaker and text
# @markdown After analyzing the file please a detailed_summary of the transcriptions
# @markdown ```
AUDIO_CONTEXT_PROMPT = "You are a transcriber specializing in transcribing Singaporean conversations. Be as accurate as possible in the transcription, give me as many rows as possible. The given audio is in a classroom setting, with a teacher and many students. Please label the teacher as Teacher, and the students as Student A Student B etc etc, and present it in a table form with the fields speaker and text After analyzing the file please a detailed_summary of the transcriptions "# @param {type:"string"}

# --- End of Prompts ---

def main():
# --- App layout ---
    tab1, tab2 = st.tabs(["Document + Audio", "Audio",])
    with tab1:
        st.subheader("Citing information in lessons to the slides", divider='gray')
        st.markdown("""
                    Gemini is able to ingest large amounts of unstructured data such as PDFs and audio
                    to help with the citations of materials. \n
                    """)

        if "text_input_value" not in st.session_state:
            st.session_state.text_input_value = ""

        button_pressed = ""

        for i, prompt in enumerate(pdf_sample_question_array):
            if st.button(prompt):
                button_pressed = prompt
                st.session_state.text_input_value=button_pressed

        question = st.text_input(
            "Input your request\n\n", key="question", value=st.session_state.text_input_value
        )

        print(f"report_question: {question}")

        uc1_reports = st.multiselect(
            "Select the slides you want to include. \n\n",
            [
                "Geography lesson - Topic 2.3 Sustainable Tourism Development",
            ],
            key="uc1_reports",
        )

        uc1_audio = st.multiselect(
            "Select the audio you would like to reference. \n\n",
            [
                "geogLesson",
            ],
            key="uc1_audio",
        )

        pdf_prompt = f"""
                    Analyze the document(s) carefully.
                    Pay attention to each subsection as they provide answers to the questions, and quote the
                    report and subsection that you referred to in your answer.
                    {question}
                    """
        # Button to Analyze Audio & Doc
        generate_t2t_uc1 = st.button("Analyze my report", key="generate_t2t_uc1")

        # Generate response from given resources when pressed
        if generate_t2t_uc1 and pdf_prompt:
            print(f"report_question: {question}")
            with st.spinner("Analyzing using Gemini 1.5 Pro ..."):
                first_tab1, first_tab2, first_tab3 = st.tabs(["Analysis", "Prompt", "Sample"])
                with first_tab1:
                    # Resources you are providing the AI Model
                    contents =[]
                    for report in uc1_reports:
                        pdf_file_uri = your_pdf_storage_prefix + report + ".pdf"
                        pdf_file = genai.upload_file(path=pdf_file_uri, mime_type="application/pdf",display_name=report)
                        contents.append(pdf_file)
                    for audio in uc1_audio:
                        audio_uri = your_pdf_audio_prefix + audio + ".MP3"
                        audio_file_uploaded = genai.upload_file(path=audio_uri, mime_type="audio/mp3",display_name=audio)
                        contents.append(audio_file_uploaded)
                    contents.append(pdf_prompt)
                    # Generate a Response from the AI Model
                    generate_ai_response(contents)

                with first_tab2:
                    st.text(pdf_prompt)
                with first_tab3:
                    st.markdown(tab_1_3_text)

    with tab2:
        st.subheader(
            "Transcribing audio with Gemini", divider='gray'
        )
        st.write(
            """
            Gemini 1.5 is able to transcribe conversations from audio. It is able to do so with great accuracy and
            is able to identify multiple languages without any pre-calibration, as compared to traditional transcribing software.
            It can return the transcript in a file format that can be further processed as seen below.
            It also has a 1 Million context window. This allows you to process up to 11h of audio in a single prompt"""
        )

        #audio_file = Part.from_uri(audio_file_uri, mime_type=audio_type)

        if audio_geolocation_url:
            st.audio(audio_geolocation_url, format=audio_type)
            tabo1, tabo2, tabo3 = st.tabs(["Response", "Prompt", "Answers"])

            audio_geolocation_description = st.button(
                "Generate", key="audio_geolocation_description"
            )

            with tabo1:
                if audio_geolocation_description and AUDIO_CONTEXT_PROMPT:
                    with st.spinner("Analyzing using Gemini 1.5 Pro ..."):
                        contents = [audio_file_tab_uploaded, AUDIO_CONTEXT_PROMPT,]
                        generate_ai_response(contents)
                        st.markdown("\n\n\n")
            with tabo2:
                st.write("Prompt used:")
                st.write(AUDIO_CONTEXT_PROMPT, "\n", "{prompt}")
            with tabo3:
                st.write("Answers:")
                st.write(
                    audio_sample_output
                )
# --- End of App layout ---

if __name__ == "__main__":
    # Remove the line that was causing the error
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(working_dir)
    main()