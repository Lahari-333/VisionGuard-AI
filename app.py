"""
VisionGuard AI - Web Application & Security Operations Center (SOC) Dashboard
Built with Streamlit, OpenCV, YOLO, and Plotly.
Original implementation - No proprietary or third-party cloned code.
"""

import os
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    PALETTE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LOITERING_THRESHOLD_SEC,
    DEFAULT_RESTRICTED_ZONE,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    SEVERITY_NORMAL,
    OUTPUTS_DIR,
    SECURITY_TARGET_CLASSES
)
from detector import VisionDetector
from tracker import VisionTracker
from alerts import AlertManager

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} | Surveillance SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom SOC Dark Theme Styling
# ---------------------------------------------------------
st.markdown(f"""
<style>
    /* Dark Navy Base Theme */
    .stApp {{
        background-color: {PALETTE['background']};
        color: {PALETTE['text']};
    }}
    
    /* Top Header Bar */
    .soc-header {{
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid {PALETTE['surface_border']};
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }}
    
    .soc-title {{
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #F8FAFC;
        margin: 0;
    }}
    
    .soc-subtitle {{
        font-size: 13px;
        color: {PALETTE['text_muted']};
        margin-top: 4px;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['surface_border']};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {PALETTE['primary']};
        transform: translateY(-2px);
    }}
    
    .metric-val {{
        font-size: 32px;
        font-weight: 700;
        margin: 6px 0;
    }}
    
    .metric-lbl {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {PALETTE['text_muted']};
    }}
    
    /* Badges */
    .badge-critical {{
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}
    
    .badge-warning {{
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}
    
    .badge-normal {{
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {{
        background-color: #0A0F1D;
        border-right: 1px solid #1E293B;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_resource
def load_detector(model_name: str = "yolov8n.pt", conf_thresh: float = 0.4):
    """Caches the YOLO detector instance for fast inference."""
    return VisionDetector(model_path=model_name, confidence_threshold=conf_thresh)


def generate_sample_surveillance_video(output_path: str, duration_sec: int = 6, fps: int = 20):
    """
    Generates an original synthetic surveillance test clip demonstrating
    a pedestrian walking into a restricted zone and loitering.
    Ensures complete functional verification even without external uploads.
    """
    width, height = 640, 360
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Zone coordinates: center 50%
    zx1, zy1, zx2, zy2 = int(width * 0.3), int(height * 0.25), int(width * 0.7), int(height * 0.75)

    for i in range(total_frames):
        # Create dark surveillance scene
        frame = np.full((height, width, 3), (25, 25, 30), dtype=np.uint8)
        
        # Grid lines (CCTV background floor grid)
        for gx in range(0, width, 40):
            cv2.line(frame, (gx, 0), (gx, height), (35, 38, 45), 1)
        for gy in range(0, height, 40):
            cv2.line(frame, (0, gy), (width, gy), (35, 38, 45), 1)

        # Draw physical perimeter markings
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (50, 60, 80), 1)
        cv2.putText(frame, "RESTRICTED AREA", (zx1 + 10, zy1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 75, 100), 1)

        # Subject 1: Walks from left to center, enters restricted area and stops (loiters)
        s1_x = min(int(50 + (i / total_frames) * 350), 320)
        s1_y = 180
        
        # Draw realistic synthetic humanoid pedestrian figure
        # Head
        cv2.circle(frame, (s1_x, s1_y - 30), 10, (210, 210, 210), -1)
        # Torso
        cv2.rectangle(frame, (s1_x - 12, s1_y - 20), (s1_x + 12, s1_y + 15), (180, 150, 100), -1)
        # Legs
        cv2.line(frame, (s1_x - 6, s1_y + 15), (s1_x - 8, s1_y + 45), (160, 130, 90), 4)
        cv2.line(frame, (s1_x + 6, s1_y + 15), (s1_x + 8, s1_y + 45), (160, 130, 90), 4)

        # Subject 2: Ambient pedestrian passing through top corridor outside restricted zone
        s2_x = (int(width - (i / total_frames) * 500)) % width
        s2_y = 60
        cv2.circle(frame, (s2_x, s2_y - 15), 6, (190, 190, 190), -1)
        cv2.rectangle(frame, (s2_x - 7, s2_y - 9), (s2_x + 7, s2_y + 12), (100, 160, 180), -1)
        cv2.line(frame, (s2_x - 4, s2_y + 12), (s2_x - 5, s2_y + 26), (90, 140, 160), 3)
        cv2.line(frame, (s2_x + 4, s2_y + 12), (s2_x + 5, s2_y + 26), (90, 140, 160), 3)

        # CCTV Timestamp watermark
        ts_text = f"CAM-04 | 2026-11-12 {time.strftime('%H:%M:%S')}.{i%20:02d}"
        cv2.putText(frame, ts_text, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        out.write(frame)

    out.release()
    return output_path


def process_video_pipeline(
    video_path: str,
    detector: VisionDetector,
    restricted_zone: List[float],
    loitering_thresh: float,
    confidence_thresh: float,
    frame_stride: int = 1,
    progress_bar=None,
    status_text=None
):
    """
    Executes the end-to-end VisionGuard AI surveillance analytics pipeline:
    Frame decoding -> YOLO detection -> Multi-object tracking -> Rule evaluation -> HUD rendering -> Video export.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video source or corrupted file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Tracker and Alert Manager
    tracker = VisionTracker(max_disappeared=15, iou_threshold=0.25, max_distance=150.0)
    alert_mgr = AlertManager(
        restricted_zone_norm=restricted_zone,
        loitering_threshold_sec=loitering_thresh,
        intrusion_target_classes=["person"]
    )

    # Output video writer
    output_filename = f"annotated_surveillance_{int(time.time())}.mp4"
    output_filepath = str(OUTPUTS_DIR / output_filename)
    
    # Try H.264 / mp4v codec for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_filepath, fourcc, fps / frame_stride, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filepath, fourcc, fps / frame_stride, (width, height))

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    preview_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        timestamp_sec = frame_idx / fps

        # 1. AI Object Detection
        raw_detections = detector.detect(frame, confidence=confidence_thresh)

        # 2. Multi-Object Identity Tracking
        tracked_subjects = tracker.update(raw_detections, timestamp_sec=timestamp_sec, frame_idx=frame_idx)

        # 3. Rule Evaluation & Threat Detection
        alert_mgr.evaluate_frame(
            active_subjects=tracked_subjects,
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
            frame_width=width,
            frame_height=height
        )

        metrics = alert_mgr.get_summary_metrics()

        # 4. Professional HUD Rendering
        annotated_frame = detector.draw_hud(
            frame=frame,
            tracked_subjects=tracked_subjects,
            restricted_zone_norm=restricted_zone,
            threat_level=metrics["threat_level"],
            fps=fps
        )

        out.write(annotated_frame)
        processed_count += 1

        # Keep periodic preview frames for UI display
        if processed_count % 10 == 0 or processed_count == 1:
            preview_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            if len(preview_frames) > 6:
                preview_frames.pop(0)

        # Update progress
        if progress_bar and total_frames > 0:
            pct = min(1.0, frame_idx / total_frames)
            progress_bar.progress(pct)
            if status_text:
                elapsed = time.time() - start_time
                proc_fps = processed_count / max(0.01, elapsed)
                status_text.markdown(
                    f"**Analyzing Frame {frame_idx}/{total_frames}** ({proc_fps:.1f} FPS) | "
                    f"Threat Level: **{metrics['threat_level']}** | "
                    f"Events Detected: **{metrics['total_events']}**"
                )

    cap.release()
    out.release()

    return {
        "output_video_path": output_filepath,
        "alert_manager": alert_mgr,
        "metrics": alert_mgr.get_summary_metrics(),
        "preview_frames": preview_frames,
        "total_frames_processed": processed_count,
        "total_time_sec": time.time() - start_time
    }


# ---------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ VISIONGUARD AI")
    st.caption("v1.0.0 | Security Operations Center")
    st.markdown("---")

    st.subheader("📹 Video Feed Source")
    feed_mode = st.radio(
        "Select Source",
        ["Upload Surveillance Video (MP4)", "Load Built-In Surveillance Demo"],
        index=0
    )

    uploaded_file = None
    if feed_mode == "Upload Surveillance Video (MP4)":
        uploaded_file = st.file_uploader(
            "Upload MP4/AVI/MOV Footage",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload recorded surveillance footage for threat analysis."
        )
        if uploaded_file is not None:
            # File validation
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 250:
                st.error(f"File size exceeds 250MB limit ({file_size_mb:.1f} MB). Please upload a shorter clip.")
                uploaded_file = None
            else:
                st.success(f"Loaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")

    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    
    conf_threshold = st.slider(
        "AI Detection Confidence",
        min_value=0.20,
        max_value=0.85,
        value=0.38,
        step=0.02,
        help="Minimum confidence required to classify target objects."
    )

    loiter_threshold = st.slider(
        "Loitering Threshold (Seconds)",
        min_value=2.0,
        max_value=20.0,
        value=DEFAULT_LOITERING_THRESHOLD_SEC,
        step=0.5,
        help="Duration a subject must remain in view before triggering a loitering warning."
    )

    stride_option = st.selectbox(
        "Analysis Sampling Rate",
        [1, 2, 3],
        index=0,
        format_func=lambda x: f"Every frame (1x - Maximum Accuracy)" if x == 1 else f"Every {x} frames ({x}x - Faster Inference)"
    )

    st.markdown("---")
    st.subheader("🛑 Restricted Perimeter Setup")
    st.caption("Normalized bounds (0.0 to 1.0 of frame dimensions)")
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        zx_min = st.slider("X Min %", 0.0, 0.8, 0.25, 0.05)
        zy_min = st.slider("Y Min %", 0.0, 0.8, 0.25, 0.05)
    with col_z2:
        zx_max = st.slider("X Max %", 0.2, 1.0, 0.75, 0.05)
        zy_max = st.slider("Y Max %", 0.2, 1.0, 0.75, 0.05)

    restricted_zone = [zx_min, zy_min, max(zx_min + 0.1, zx_max), max(zy_min + 0.1, zy_max)]

    st.markdown("---")
    run_btn = st.button("🚀 Run Surveillance Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------------
# Top Header
st.markdown(f"""
<div class="soc-header">
    <div>
        <div class="soc-title">🛡️ {APP_NAME}</div>
        <div class="soc-subtitle">{APP_TAGLINE}</div>
    </div>
    <div style="text-align: right;">
        <span class="badge-normal">● AI ENGINE ONLINE</span>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">LOCAL EDGE INFERENCE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_live, tab_events, tab_analytics, tab_architecture = st.tabs([
    "🖥️ SOC Surveillance & Analysis",
    "🚨 Security Incident Log",
    "📊 Threat Intelligence Analytics",
    "⚙️ System Architecture & Legal"
])


# Session State Management
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Video Processing Trigger
if run_btn:
    temp_input_video = None
    
    if feed_mode == "Upload Surveillance Video (MP4)":
        if uploaded_file is None:
            st.error("Please upload a video file or switch to 'Load Built-In Surveillance Demo'.")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            temp_input_video = tfile.name
    else:
        # Generate demo synthetic video
        demo_path = str(OUTPUTS_DIR / "demo_surveillance_feed.mp4")
        with st.spinner("Generating synthetic surveillance scenario..."):
            generate_sample_surveillance_video(demo_path, duration_sec=7, fps=20)
        temp_input_video = demo_path

    if temp_input_video:
        detector = load_detector("yolov8n.pt", conf_thresh=conf_threshold)

        progress_container = st.container()
        with progress_container:
            p_bar = st.progress(0.0)
            status_txt = st.empty()

        try:
            with st.spinner("VisionGuard AI is actively processing video frames..."):
                results = process_video_pipeline(
                    video_path=temp_input_video,
                    detector=detector,
                    restricted_zone=restricted_zone,
                    loitering_thresh=loiter_threshold,
                    confidence_thresh=conf_threshold,
                    frame_stride=stride_option,
                    progress_bar=p_bar,
                    status_text=status_txt
                )
            st.session_state.analysis_results = results
            st.success("✅ Surveillance video analysis completed successfully!")
        except Exception as ex:
            st.error(f"Analysis Error: {str(ex)}")


# ---------------------------------------------------------
# TAB 1: SOC Surveillance & Video Analysis
# ---------------------------------------------------------
with tab_live:
    results = st.session_state.analysis_results

    if results is not None:
        metrics = results["metrics"]
        
        # Threat Level Indicator
        t_level = metrics["threat_level"]
        badge_class = "badge-critical" if t_level == "CRITICAL" else \
                      "badge-warning" if t_level == "ELEVATED" else "badge-normal"

        # KPI Metrics Cards Row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Detections</div>
                <div class="metric-val" style="color: #60A5FA;">{metrics['total_detections']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">People Detected</div>
                <div class="metric-val" style="color: #38BDF8;">{metrics['unique_people']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Vehicles Detected</div>
                <div class="metric-val" style="color: #A78BFA;">{metrics['unique_vehicles']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Zone Intrusions</div>
                <div class="metric-val" style="color: #F87171;">{metrics['intrusion_events']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Loitering Violations</div>
                <div class="metric-val" style="color: #FBBF24;">{metrics['loitering_events']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Operational Threat</div>
                <div class="metric-val"><span class="{badge_class}">{t_level}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Video Player & Inspection
        v_col1, v_col2 = st.columns([1.6, 1])

        with v_col1:
            st.subheader("🎥 Annotated Surveillance Feed (HUD)")
            out_vid_path = results["output_video_path"]

            if os.path.exists(out_vid_path):
                # Check video file size
                v_size = os.path.getsize(out_vid_path)
                if v_size > 0:
                    with open(out_vid_path, "rb") as vf:
                        st.video(vf.read(), format="video/mp4")
                else:
                    st.warning("Video output stream is empty.")

            # Download button for processed video
            with open(out_vid_path, "rb") as f_down:
                st.download_button(
                    label="📥 Download Annotated Surveillance Video (MP4)",
                    data=f_down,
                    file_name=os.path.basename(out_vid_path),
                    mime="video/mp4",
                    use_container_width=True
                )

        with v_col2:
            st.subheader("⚡ Real-Time Security Incident Feed")
            alert_mgr = results["alert_manager"]
            events = alert_mgr.events

            if len(events) == 0:
                st.info("No security incidents detected. Perimeter is secure.")
            else:
                for ev in reversed(events[-6:]):  # Show latest 6 events
                    badge_style = "badge-critical" if ev.severity == SEVERITY_CRITICAL else \
                                  "badge-warning" if ev.severity == SEVERITY_WARNING else "badge-normal"
                    st.markdown(f"""
                    <div style="background-color: {PALETTE['surface']}; border-left: 4px solid {'#EF4444' if ev.severity == SEVERITY_CRITICAL else '#F59E0B'}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="{badge_style}">{ev.severity.upper()}</span>
                            <span style="font-size: 11px; color: {PALETTE['text_muted']};">⏱️ {ev.timestamp_str}</span>
                        </div>
                        <div style="font-size: 13px; font-weight: 600; margin-top: 6px; color: #E2E8F0;">
                            {ev.event_type} (Track #{ev.track_id})
                        </div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 3px;">
                            {ev.description}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Frame Snapshots Gallery
        if results["preview_frames"]:
            st.markdown("---")
            st.subheader("📸 Detection Keyframes")
            preview_cols = st.columns(len(results["preview_frames"]))
            for idx, p_frame in enumerate(results["preview_frames"]):
                with preview_cols[idx]:
                    st.image(p_frame, caption=f"Capture #{idx+1}", use_container_width=True)

    else:
        # Empty State
        st.info("👈 Upload surveillance footage in the sidebar and click **'Run Surveillance Analysis'** to begin.")
        
        # Display perimeter setup preview
        st.subheader("🎯 Restricted Perimeter Geometry Preview")
        preview_bg = np.full((300, 600, 3), (20, 24, 34), dtype=np.uint8)
        # Grid lines
        for gx in range(0, 600, 40):
            cv2.line(preview_bg, (gx, 0), (gx, 300), (30, 36, 48), 1)
        for gy in range(0, 300, 40):
            cv2.line(preview_bg, (0, gy), (600, gy), (30, 36, 48), 1)

        # Draw zone
        px1 = int(restricted_zone[0] * 600)
        py1 = int(restricted_zone[1] * 300)
        px2 = int(restricted_zone[2] * 600)
        py2 = int(restricted_zone[3] * 300)

        zone_preview = preview_bg.copy()
        cv2.rectangle(zone_preview, (px1, py1), (px2, py2), (30, 140, 240), -1)
        cv2.addWeighted(zone_preview, 0.25, preview_bg, 0.75, 0, preview_bg)
        cv2.rectangle(preview_bg, (px1, py1), (px2, py2), (30, 140, 240), 2)
        cv2.putText(preview_bg, "RESTRICTED SECURITY ZONE", (px1 + 8, py1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        st.image(preview_bg, caption="Active Perimeter Configuration", use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Security Incident Log & CSV Export
# ---------------------------------------------------------
with tab_events:
    st.subheader("🚨 Security Incident Register")
    
    if results is not None:
        alert_mgr = results["alert_manager"]
        df_events = alert_mgr.to_dataframe()

        if df_events.empty:
            st.info("Zero incidents recorded for this surveillance session.")
        else:
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                severity_filter = st.selectbox(
                    "Filter by Severity",
                    ["All Severities", SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_NORMAL]
                )

            filtered_df = df_events
            if severity_filter != "All Severities":
                filtered_df = df_events[df_events["Severity"] == severity_filter]

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            # Export CSV Button
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Security Incident Report (CSV)",
                data=csv_data,
                file_name=f"visionguard_incident_report_{int(time.time())}.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.info("Run surveillance analysis to populate the security incident register.")


# ---------------------------------------------------------
# TAB 3: Threat Intelligence Analytics (Plotly)
# ---------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Surveillance Threat Analytics")

    if results is not None:
        alert_mgr = results["alert_manager"]
        df_events = alert_mgr.to_dataframe()
        metrics = results["metrics"]

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            # 1. Classification distribution
            cat_data = {
                "Category": ["People", "Vehicles"],
                "Count": [metrics["unique_people"], metrics["unique_vehicles"]]
            }
            fig_cat = px.pie(
                cat_data,
                names="Category",
                values="Count",
                title="<b>Tracked Subject Classification</b>",
                hole=0.55,
                color_discrete_sequence=["#3B82F6", "#8B5CF6"]
            )
            fig_cat.update_layout(
                paper_bgcolor=PALETTE["surface"],
                plot_bgcolor=PALETTE["surface"],
                font_color="#F3F4F6",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with chart_c2:
            # 2. Alert Severity Breakdown
            sev_data = {
                "Severity": ["Critical (Intrusion)", "Warning (Loitering)"],
                "Incidents": [metrics["critical_alerts"], metrics["warning_alerts"]]
            }
            fig_sev = px.bar(
                sev_data,
                x="Severity",
                y="Incidents",
                title="<b>Threat Alert Distribution</b>",
                color="Severity",
                color_discrete_map={
                    "Critical (Intrusion)": "#EF4444",
                    "Warning (Loitering)": "#F59E0B"
                }
            )
            fig_sev.update_layout(
                paper_bgcolor=PALETTE["surface"],
                plot_bgcolor=PALETTE["surface"],
                font_color="#F3F4F6",
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_sev, use_container_width=True)

        # 3. Incident Timeline
        st.subheader("⏱️ Security Events Temporal Timeline")
        if not df_events.empty and len(alert_mgr.events) > 0:
            timeline_records = []
            for ev in alert_mgr.events:
                timeline_records.append({
                    "Timestamp (s)": ev.timestamp_sec,
                    "Event Type": ev.event_type,
                    "Severity": ev.severity,
                    "Track ID": f"Subject #{ev.track_id}"
                })
            df_tl = pd.DataFrame(timeline_records)

            fig_tl = px.scatter(
                df_tl,
                x="Timestamp (s)",
                y="Event Type",
                color="Severity",
                symbol="Severity",
                hover_data=["Track ID"],
                size_max=14,
                title="<b>Chronological Security Breaches & Violations</b>",
                color_discrete_map={
                    SEVERITY_CRITICAL: "#EF4444",
                    SEVERITY_WARNING: "#F59E0B",
                    SEVERITY_NORMAL: "#10B981"
                }
            )
            fig_tl.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
            fig_tl.update_layout(
                paper_bgcolor=PALETTE["surface"],
                plot_bgcolor=PALETTE["surface"],
                font_color="#F3F4F6",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("No timeline events to display.")

    else:
        st.info("Run surveillance analysis to generate threat intelligence charts.")


# ---------------------------------------------------------
# TAB 4: System Architecture & Compliance
# ---------------------------------------------------------
with tab_architecture:
    st.subheader("⚙️ VisionGuard AI System Architecture")
    
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Input["Surveillance Input"]
            V["Video Feed (MP4/CCTV)"]
        end

        subgraph Engine["VisionGuard AI Processing Core"]
            D["YOLOv8 Edge Neural Detector"]
            T["Centroid & IoU Multi-Subject Tracker"]
            A["Rules Engine & Alert Manager"]
        end

        subgraph Analysis["Security Intelligence"]
            Z["Restricted Perimeter Breach Detection"]
            L["Dwell Duration & Loitering Analysis"]
        end

        subgraph Output["SOC Operator Interface"]
            HUD["Annotated Video Stream (HUD)"]
            CSV["Security Incident Export (CSV)"]
            Dash["Interactive Threat Analytics"]
        end

        V --> D
        D -->|Bounding Boxes & Confidences| T
        T -->|Persistent Subject Trajectories| A
        A --> Z
        A --> L
        Z -->|Critical Alert| HUD
        L -->|Warning Alert| HUD
        A --> CSV
        A --> Dash
    ```
    """)

    st.markdown("---")
    st.subheader("📜 Legal Compliance, Privacy & Dependency Attributions")
    
    st.markdown("""
    ### Privacy & Edge Execution Guarantee
    VisionGuard AI processes all video footage **100% locally on device**. No video frames, biometric data, 
    or operational telemetry are transmitted to third-party cloud services or external servers.

    ### Dependency Licenses & Attributions
    - **Ultralytics YOLOv8**: Licensed under AGPL-3.0 (with commercial licensing options available directly from Ultralytics Inc.). Used for object detection inference.
    - **OpenCV (Open Source Computer Vision Library)**: Licensed under Apache 2.0. Used for video decoding, image transformation, and visual overlay drawing.
    - **Streamlit**: Licensed under Apache 2.0. Used for user interface rendering and state management.
    - **Plotly**: Licensed under MIT License. Used for interactive data visualizations and timeline charting.
    - **Pandas**: Licensed under BSD 3-Clause. Used for event log aggregation and tabular data export.

    ### Originality Statement
    *VisionGuard AI is an independently designed, architected, and coded smart video surveillance system. 
    It does not replicate, clone, or reuse any third-party proprietary source code or assets.*
    """)
