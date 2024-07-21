"""
Code to call summarizer model, and to build streamlit app to take in necessry input.
"""

#calling previously built summarizer model
summarizer = pipeline("summarization", model="scural/arxiv_model")

#utilizing youtube api to get existing transcripts of videos
!pip install youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from pdfquery import PDFQuery
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

import streamlit as st

"""Importing Model To Streamlit"""

!pip install -q streamlit
!pip install pdfquery
!pip install transformers

%%writefile app.py #build streamlit app
st.title("Bullet-Point Generator")

#helper functions to clean up input data
def generate_key_points(text):
  summarization_pipeline = pipeline(task="summarization", model="scural/arxiv_model")
  text = text.split(" ")
  bullet_points = []

  if len(text) > 200:
    iteration = len(text)//200
    k = 0

    for i in range(iteration):
      chunk = text[k:200+k]
      point = summarization_pipeline("summarize: " + ' '.join(chunk))
      #bullet_points = bullet_points + ' \n' + point[0].get('summary_text')
      bullet_points.append(point[0].get('summary_text'))
      k += 200

    return bullet_points

  summaries = summarization_pipeline("summarize: " + " ".join(text))
  return summaries[0].get('summary_text')

#helper function to extract summary
def extract_data(feed):
  pdf = PDFQuery(feed)
  pdf.load()
  text_elements = pdf.pq('LTTextLineHorizontal')

  # Extract the text from the elements
  text = [t.text for t in text_elements]

  text = " ".join(text)
  text.replace('\n', '')

  return text

#helper function to extract transcript
def extract_transcript(link):
  link = link.replace("https://www.youtube.com/watch?v=", "")
  yt_id = link[:link.index("_")]
  srt = YouTubeTranscriptApi.get_transcript(yt_id)
  transcript = ""
  for i in srt:
    transcript = transcript + i['text']
  return transcript

#FOR PDF INPUT
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
  st.success("Extracting Bullet-Points!")
  text = extract_data(uploaded_file)
  points = generate_key_points(text)

  if type(points) == list:
    for point in points:
      st.write("*" + point)
      #st.write("Bullet points: ", point)

  else: st.write(points)

#FOR TEXT INPUT
text_input = st.text_input(
  "Enter some text",
  label_visibility="visible",
  disabled=False,)

if text_input:
  st.success("Extracting Bullet-Points!")
  points = generate_key_points(text_input)

  if type(points) == list:
    for point in points:
      #st.write("Bullet points: ", point)
      st.write("*" + point)

  else: st.write(points)

#FOR VIDEO INPUT
vid_input = st.text_input(
  "Or add a YouTube link",
  label_visibility="visible",
  disabled=False,
)

if vid_input:
  st.success("Extracting Video Bullet-points!")
  transcript = extract_transcript(vid_input)
  points = generate_key_points(transcript)

  if type(points) == list:
    for point in points:
      st.write("*" + point)
      #st.write("Bullet points: ", point)

  else: st.write(points)

#generate local streamlit app
!npm install localtunnel
!streamlit run /content/app.py &>/content/logs.txt &
!npx localtunnel --port 8501 & curl ipv4.icanhazip.com
