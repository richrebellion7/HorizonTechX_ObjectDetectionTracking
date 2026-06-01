import streamlit as st
import cv2
import tempfile
from collections import Counter
from ultralytics import YOLO
import os
import numpy as np

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎯 AI-Powered Object Detection & Tracking System")

st.markdown(
    """
    Upload images or videos and perform object detection and tracking using YOLOv8.
    """
)


st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)


mode = st.radio(
    "Select Input Type",
    [
        "Image",
        "Video"
    ]
)
if mode == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image_bytes = uploaded_image.read()

        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        results = model.track(
            image_array,
            persist=True,
            conf=confidence,
            verbose=False
        )

        annotated_image = results[0].plot()

        st.image(
            annotated_image,
            caption="Detected & Tracked Image",
            use_column_width=True
        )

if mode == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    st.warning(
    """
    ⚠️ Large videos require significant processing time.

    Processing speed depends on:
    - Video length
    - Resolution
    - Number of detectable objects
    - System performance

    Videos longer than 2 minutes may take a few minutes to complete. The application remains active during processing, so feel free to use other applications while you wait.
    """
    )

    if uploaded_video:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())

        video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)

        output_path = "processed_output.mp4"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        current_frame = 0

        object_counts = Counter()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                verbose=False
            )

            annotated_frame = results[0].plot()

            video_writer.write(annotated_frame)

            if results[0].boxes is not None:

                for cls_id in results[0].boxes.cls.tolist():

                    class_name = model.names[int(cls_id)]

                    object_counts[class_name] += 1

            current_frame += 1

            progress = min(
                current_frame / frame_count,
                1.0
            )

            progress_bar.progress(progress)

            status_text.info(
                f"Processing Video... {int(progress * 100)}%"
            )

        cap.release()
        video_writer.release()

        status_text.empty()

        st.success("Processing Complete")

        st.subheader("Processed Video")

        st.video(output_path)

        with open(output_path, "rb") as file:

            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="tracked_video.mp4",
                mime="video/mp4"
            )

        st.subheader("Detection Summary")

        if object_counts:

            cols = st.columns(
                min(len(object_counts), 4)
            )

            for i, (obj, count) in enumerate(
                object_counts.items()
            ):

                cols[i % 4].metric(
                    label=obj.title(),
                    value=count
                )

        else:

            st.info(
                "No objects detected."
            )
