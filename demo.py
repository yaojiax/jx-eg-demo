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

PROJECT_ID = "jiaxin-project"  # Your Google Cloud Project ID
LOCATION = "us-central1"  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
header_text = "Test"
header_img_link = "https://storage.mtls.cloud.google.com/jx-demo-placeholder-media/header_image~2.png"
reports_string_array = [
            "RB1_General_Safety",
            "RB10_Exceptional_Situations",
            "RB11_Signalling",
]
your_pdf_storage_prefix = "gs://smrtexcel/"
pdf_sample_question_array = [
        "How can a staff member gain access to the track?",
        "What happens if there is an air burst in the rear EMU?",
        "The train in automatic mode is not automatically proceeding when the route signal is cleared. What should I do?"
    ] # Takes in as many as you want

pdf_prompt = f"""
                Your are a very professional document summarization specialist and you pay key attention to the details
                Look through the whole document carefully
                Please answer the following questions, and highlight any assumptions made:
                """
audio_geolocation_url = (
        "https://storage.mtls.cloud.google.com/zysmrt-media/incident.mp3"
    )

audio_file_uri = "gs://zysmrt-media/incident.mp3"
audio_type = "audio/mp3"
audio_prompt = """
                    You are a transcriber specializing in transcribing Singaporean conversations.
                    Be as accurate as possible in the transcription, give me as many rows as possible.
                    The given audio is an argument between 2 commuters on a train. 
                    Please label the speaker as person A and person B, and present it in a table form
                    with the fields speaker and text
                    After analyzing the file please a detailed_summary of the argument
                """
audio_sample_output= """
           | Speaker | Text |
           |---|---|
           | Person A | 为什么我不可以讲呢？ (Wèishénme wǒ bù kěyǐ jiǎng ne?) |
           | Person B | What's your problem? |
           | Person A | What's your problem? |
           | Person B | I said to the old senior people, I said can you give your seat to the old people? |
           | Person A | 啊，这里没有人要坐。(Ā, zhèli méiyǒu rén yào zuò.) |
           | Person B | I said please can you? You just say please, just now. You cannot be rude to people. I never ask you, please, because you are youngster. |
           | Person A | I am also woman. <noise> 你不可以乱乱骂。我刚才讲了我肚子痛，我没有听到，你还要讲。 你还是欧巴桑，要懂得-- |
           | Person B | You see, but you have nothing like this. Why you want to lie? |
           | Person A | 我没有骗你。(Wǒ méiyǒu piàn nǐ.) |
           | Person B | You are Singaporean, you lose the face of Singapore, you know? You lose the, you lose the face of Singapore, you know? Lose the face. You lose the face, yes. You Singapore-- |
            \n
            Detailed Summary
            This audio clip captures a heated argument between two commuters (Person A and Person B) on a Singaporean train.

            The argument stems from Person B asking Person A to give up their seat for an elderly passenger. Person B claims they politely asked for the seat ("I said please can you?"), while Person A argues they were not informed of the request ("啊，你没有告诉我！").

            The situation escalates quickly with both parties exchanging accusations. Person B criticizes Person A's behavior, suggesting they are being disrespectful and are giving Singaporeans a bad name ("You think you are Singaporean, you lose the face of Singapore, you know?"). Person A, feeling wrongly accused, defends themselves by stating they were not being impolite ("我没有不礼貌！").

            The audio ends abruptly, leaving the resolution of the argument unknown. However, the conversation highlights the sensitivity surrounding respect for the elderly on public transport in Singapore and the potential for misunderstandings to escalate quickly.    
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
    st.subheader("Creating a QnA Chatbot to give replies based on rulebooks", divider='gray')
    st.markdown("""
                Gemini is able to ingest large amounts of unstructured data such as PDFs and become a search engine
                for questions related to the PDFs, such as the rulebook. It is also able to cite where exactly
                the information is from, to allow for quick validation. \n
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
        "Select the reports you want to include. \n\n",
        [
            "RB1_General_Safety",
            "RB10_Exceptional_Situations",
            "RB11_Signalling"
        ],
        key="uc1_reports",
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
            first_tab1, first_tab2 = st.tabs(["Analysis", "Prompt"])
            with first_tab1:
                #model = GenerativeModel(model_name="gemini-1.5-pro-001")
                contents =[]
                for report in uc1_reports:
                    pdf_file_uri = your_pdf_storage_prefix + report + ".pdf"
                    print(pdf_file_uri)
                    pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
                    contents.append(pdf_file)
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
