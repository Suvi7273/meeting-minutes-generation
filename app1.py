
import os
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForTokenClassification

from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import cv2
import numpy as np
import dlib

face_detector = dlib.get_frontal_face_detector()
face_landmark_predictor = dlib.shape_predictor(r"C:\Users\ramas\OneDrive\Documents\Downloads\generate_mm\shape_predictor_68_face_landmarks_GTX.dat")
face_embedder = dlib.face_recognition_model_v1(r"C:\Users\ramas\OneDrive\Documents\Downloads\generate_mm\dlib_face_recognition_resnet_model_v1.dat")

# Set parameters for face detection
THRESHOLD = 0.5  # Similarity threshold for unique faces
FRAME_SKIP = 48  # Process every nth frame
PADDING = 20  # Bounding box expansion

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_data
def extract_audio(video_file, output_audio_file):
    try:
        # Extract audio from video
        video_clip = VideoFileClip(video_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_file, bitrate='256k')

        # Normalize the audio
        audio = AudioSegment.from_file(output_audio_file)
        normalized_audio = audio.apply_gain(-audio.max_dBFS)  # Normalize to 0 dBFS
        normalized_audio.export(output_audio_file, format="wav")

        return output_audio_file
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None


@st.cache_data
def transcribe_audio(audio_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v2"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Get forced decoder IDs for English transcription and set them directly in model config
    #model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids

    # Create the pipeline without passing forced_decoder_ids directly
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    # Run transcription with the language already set in the model's configuration
    result = pipe(audio_file)
    print(result["text"])
    return result["text"]


@st.cache_data
def summarize_text(text):
    x = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summary_text = x[0]['summary_text']  # Extract the summary text from the dictionary
    print(summary_text)
    return summary_text
    
@st.cache_data
def ner(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    person_entities = set([entity['word'] for entity in ner_results if entity['entity'] in ['B-PER', 'I-PER']])

    print(person_entities)
    return person_entities

def generate_meeting_minutes(date, time, location, present, absent, reports, additions, adjournment_time, adjournment_name, next_meeting):
    minutes = f"""
    Meeting Minutes
    Date: {date}
    ________________________________________
    Opening:
    The meeting was called to order at {time} by {adjournment_name} at {location}.
    
    Present:
    {', '.join(present)}
    
    Absent:
    {', '.join(absent)}
    
    Approval of Agenda:
    The agenda was reviewed and approved.
    
    Approval of Minutes:
    The minutes from the previous meeting were reviewed and approved.
    
    Reports:
    - {reports}
    
    Additions to the Agenda:
    - {additions}
    
    Adjournment:
    The meeting was adjourned at {adjournment_time} by {adjournment_name}. The next meeting will be held on {next_meeting}.
    
    Minutes submitted by:
    {adjournment_name}
    
    Minutes approved by:
    {adjournment_name}
    """
    return minutes


def process_video(video_path,audio_file):
    unique_faces = []
    unique_face_images = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.write("Error: Unable to open video file. Check file path.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("End of video reached or unable to read frame.")
            break

        # Process every nth frame based on FRAME_SKIP
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = face_detector(rgb_frame, 1)

        # Face detection fallback if no introduction is detected
        for d in dets:
            width = d.right() - d.left()
            height = d.bottom() - d.top()
            if width < 36 or height < 36:
                continue

            shape = face_landmark_predictor(rgb_frame, d)
            face_embedding = np.array(face_embedder.compute_face_descriptor(rgb_frame, shape))
            
            is_unique = True
            for unique_face in unique_faces:
                if np.linalg.norm(face_embedding - unique_face) < THRESHOLD:
                    is_unique = False
                    break

            if is_unique:
                unique_faces.append(face_embedding)
                startX, startY, endX, endY = d.left(), d.top(), d.right(), d.bottom()
                startX = max(0, startX - PADDING)
                startY = max(0, startY - PADDING)
                endX = min(frame.shape[1], endX + PADDING)
                endY = min(frame.shape[0], endY + PADDING)
                unique_face_img = frame[startY:endY, startX:endX]
                unique_face_images.append(cv2.cvtColor(unique_face_img, cv2.COLOR_BGR2RGB))

        frame_count += 1

    cap.release()
    st.session_state.unique_face_images = unique_face_images


def main():
    st.title("Meeting Minutes Generation")

    video_file = st.text_input("Provide path to the video file")
    
    audio_file = "extracted_audio.wav"
    if video_file:
        if not os.path.exists(audio_file):
            audio_file = extract_audio(video_file, audio_file)
    
    if video_file and audio_file:
        transcription = transcribe_audio(audio_file)
        
    #pre=[]
    pre=ner(transcription)

    if len(pre)==0:
        if 'unique_face_images' not in st.session_state:
            st.session_state.unique_face_images = []
        if 'present' not in st.session_state:
            st.session_state.present = []
    
        if video_file and not st.session_state.unique_face_images:
            process_video(video_file,audio_file)
            #st.success("Video processing complete!")

        if st.session_state.unique_face_images:
            st.subheader("Faces Detected:")
            for i, face_img in enumerate(st.session_state.unique_face_images):
                st.image(face_img, caption=f"Face {i + 1}", use_column_width=False, width=200)
                input_value = st.text_input(f"Input for Face {i + 1}:", "")
                if input_value != '-' and input_value and input_value not in st.session_state.present:
                    st.session_state.present.append(input_value)

    if video_file:
        if audio_file:
            summary = summarize_text(transcription)

            date = st.text_input("Date")
            time = st.text_input("Time of Meeting")
            location = st.text_input("Location")
            absent = st.text_input("Absent Members (comma-separated)").split(',')
            additions = st.text_input("Additions to the Agenda")
            adjournment_time = st.text_input("Adjournment Time")
            adjournment_name = st.text_input("Adjourned By")
            next_meeting = st.text_input("Next Meeting Date and Location")
            
            temp = pre if pre is len(pre)>0 else st.session_state.present
            
            if st.button("Generate Meeting Minutes"):
                minutes = generate_meeting_minutes(
                    date, time, location, temp, absent,
                    summary, 
                    additions, adjournment_time, adjournment_name, next_meeting)
                st.write("Generated Meeting Minutes:")
                st.text_area("Meeting Minutes", minutes, height=500)

if __name__ == "__main__":
    main()
