import pandas as pd
import os
import streamlit as st
import vertexai

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

#Change the variables here as required!

PROJECT_ID = os.environ.get("jiaxin-project")  # Your Google Cloud Project ID
LOCATION = os.environ.get("us-central1")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
header_text = "Demo"
header_img_link = "https://storage.mtls.cloud.google.com/jx-eg-demo/enggood.png"
report_string_array = [
            "Geography lesson - Topic 2.3 Sustainable Tourism Development",
]
your_pdf_storage_prefix = "gs://jx-eg-demo/"
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

audio_file_uri = "gs://jx-eg-demo/Geography lesson - audio.MP3"
audio_type = "audio/mp3"
audio_prompt = """
                    You are a transcriber specializing in transcribing Singaporean conversations.
                    Be as accurate as possible in the transcription, give me as many rows as possible.
                    The given audio is in a classroom setting, with a teacher and many students.
                    Please label the teacher as Teacher, and the students as Student A Student B etc etc, and present it in a table form
                    with the fields speaker and text
                    After analyzing the file please a detailed_summary of the transcriptions
                """
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

# change prompts here as follow

# Logic as below!

@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    multimodal_model_flash = GenerativeModel("gemini-1.5-flash-001")
    multimodal_model_pro = GenerativeModel("gemini-1.5-pro-001")
    return multimodal_model_flash, multimodal_model_pro

st.image(header_img_link)
st.header(header_text, divider="rainbow")
multimodal_model_flash, multimodal_model_pro = load_models()

# change titles here as follow

tab1, tab2 = st.tabs(
    ["Document + Audio", "Audio",]
)

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
            "Geography lesson - audio",
        ],
        key="uc1_audio",
    )

    pdf_prompt = f"""
                Analyze the document(s) carefully. 
                Pay attention to each subsection as they provide answers to the questions, and quote the 
                report and subsection that you referred to in your answer.
                {question}
                """
   
    generate_t2t_uc1 = st.button("Analyze my report", key="generate_t2t_uc1")
    if generate_t2t_uc1 and pdf_prompt:
        print(f"report_question: {question}")
        with st.spinner("Analyzing using Gemini 1.5 Pro ..."):
            first_tab1, first_tab2, first_tab3 = st.tabs(["Analysis", "Prompt", "Sample"])
            with first_tab1:
                #model = GenerativeModel(model_name="gemini-1.5-pro-001")
                contents =[]
                for report in uc1_reports:
                    pdf_file_uri = your_pdf_storage_prefix + report + ".pdf"
                    print(pdf_file_uri)
                    pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
                    contents.append(pdf_file)
                for audio in uc1_audio:
                    audio_uri = your_pdf_storage_prefix + audio + ".MP3"
                    print(audio_uri)
                    audio_file = Part.from_uri(audio_uri, mime_type=audio_type)
                    contents.append(audio_file)
                contents.append(pdf_prompt)
                generation_config = {
                        "max_output_tokens": 8192,
                        "temperature": 0.05,
                        "top_p": 0.95,
                    }               
                
                response = multimodal_model_pro.generate_content(contents,
                                                  generation_config=generation_config
                 #                                 stream=True
                                                  )
                if response:
                    st.write("Response:")
                 #   for chunk in response:
                 #       st.text(chunk.text)
                    
                    st.write(response.text)
            with first_tab2:
                st.text(pdf_prompt)
            with first_tab3:
                st.markdown(
                    """
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
                )


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

    ###
    ### EDIT HERE
    ###

    audio_file = Part.from_uri(audio_file_uri, mime_type=audio_type)

    if audio_geolocation_url:
        st.audio(audio_geolocation_url, format=audio_type)
        tabo1, tabo2, tabo3 = st.tabs(["Response", "Prompt", "Answers"])

        audio_geolocation_description = st.button(
            "Generate", key="audio_geolocation_description"
        )
        with tabo1:
            if audio_geolocation_description and audio_prompt:
                with st.spinner("Analyzing using Gemini 1.5 Pro ..."):
                    # model = GenerativeModel(model_name="gemini-1.5-pro-001")

                    generation_config = {
                        "max_output_tokens": 8192,
                        "temperature": 0.1,
                        "top_p": 0.95,
                    }


                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    }

                    
                    contents = [audio_file, audio_prompt,]
                    response = multimodal_model_pro.generate_content(contents, 
                                                      generation_config=generation_config,
                                                      safety_settings=safety_settings
                                                      #stream=True
                                                      )
                    #print(response.text)
                    #for chunk in response:
                        #    st.text(chunk.text)
                    st.markdown(response.text)
                    print(response)
                    st.markdown("\n\n\n")
        with tabo2:
            st.write("Prompt used:")
            st.write(audio_prompt, "\n", "{prompt}")
        with tabo3:
            st.write("Answers:")
            st.write(
                audio_sample_output
            )
