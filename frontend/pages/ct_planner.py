"""
CT Planner Page - With Video Upload and Similarity Detection.

Features:
1. Upload actual video files (MP4)
2. Upload media plan spreadsheet
3. Auto-match videos to media plan
4. Detect similar/duplicate videos
5. Prioritize and select creatives for testing
6. Save and load plans with persistence
"""

import streamlit as st
from datetime import date, timedelta
import uuid
import sys
import os
import tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rules_engine import get_rules_engine
from models import KPIType, Channel, AssetType

# Try to import video ingestion service
try:
    from services.video_ingestion import VideoIngestionService, create_sample_media_plan
    VIDEO_INGESTION_AVAILABLE = True
except ImportError:
    VIDEO_INGESTION_AVAILABLE = False

# Try to import persistence service
try:
    from services.persistence import get_persistence_service
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False


def render_planner():
    """Render the CT Planner page."""
    
    # Sidebar for saved plans
    if PERSISTENCE_AVAILABLE:
        _render_saved_plans_sidebar()
    
    st.markdown("## 📋 Creative Testing Planner")
    st.markdown("Upload your media plan and videos to automatically generate an optimized test plan.")
    
    # Initialize planner state
    if "planner_step" not in st.session_state:
        st.session_state.planner_step = "media_plan"
    if "plan_data" not in st.session_state:
        st.session_state.plan_data = {}
    if "selected_creatives" not in st.session_state:
        st.session_state.selected_creatives = {}
    if "video_service" not in st.session_state:
        st.session_state.video_service = None
    
    # Handle legacy step names from old code
    legacy_step_map = {
        "input": "media_plan",
        "campaign": "media_plan",
        "creatives": "videos",
        "review": "select",
        "approved": "media_plan",  # Reset if approved
    }
    if st.session_state.planner_step in legacy_step_map:
        st.session_state.planner_step = legacy_step_map[st.session_state.planner_step]
    
    # Progress indicator - Updated steps (4 steps now)
    steps = ["Upload Media Plan", "Upload Videos", "Review Matches", "Select & Approve"]
    step_keys = ["media_plan", "videos", "review_matches", "select"]
    
    # Ensure current step is valid
    if st.session_state.planner_step not in step_keys:
        st.session_state.planner_step = "media_plan"
    
    current_step_idx = step_keys.index(st.session_state.planner_step)
    
    cols = st.columns(4)
    for i, step in enumerate(steps):
        with cols[i]:
            if i < current_step_idx:
                st.success(f"✓ {step}")
            elif i == current_step_idx:
                st.info(f"→ {step}")
            else:
                st.markdown(f"○ {step}")
    
    st.markdown("---")
    
    # Render current step
    if st.session_state.planner_step == "media_plan":
        render_media_plan_upload()
    elif st.session_state.planner_step == "videos":
        render_video_upload()
    elif st.session_state.planner_step == "review_matches":
        render_match_review()
    elif st.session_state.planner_step == "select":
        render_selection_and_approval()


def _render_saved_plans_sidebar():
    """Render sidebar with saved plans."""
    with st.sidebar:
        st.markdown("### 📂 Saved Plans")
        
        try:
            persistence = get_persistence_service()
            
            # List all plans
            plans = persistence.list_plans()
            
            if plans:
                # Group by status
                approved_plans = [p for p in plans if p.get('is_approved')]
                draft_plans = [p for p in plans if not p.get('is_approved')]
                
                if approved_plans:
                    st.markdown("**✅ Approved Plans**")
                    for plan in approved_plans[:5]:  # Show last 5
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"{plan.get('campaign_name', 'Unknown')}")
                            st.caption(f"{plan.get('creative_count', 0)} creatives")
                        with col2:
                            if st.button("📂", key=f"load_approved_{plan['plan_id']}", help="Load this plan"):
                                _load_plan_from_persistence(plan['plan_id'], is_approved=True)
                
                if draft_plans:
                    st.markdown("**📝 Draft Plans**")
                    for plan in draft_plans[:5]:  # Show last 5
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"{plan.get('campaign_name', 'Unknown')}")
                        with col2:
                            if st.button("📂", key=f"load_draft_{plan['plan_id']}", help="Load this draft"):
                                _load_plan_from_persistence(plan['plan_id'], is_approved=False)
                
                st.markdown("---")
            else:
                st.caption("No saved plans yet.")
            
            # Start new plan button
            if st.button("➕ New Plan", use_container_width=True):
                st.session_state.planner_step = "media_plan"
                st.session_state.plan_data = {}
                st.session_state.selected_creatives = {}
                st.session_state.video_service = None
                st.rerun()
                
        except Exception as e:
            st.caption(f"Could not load saved plans: {e}")


def _load_plan_from_persistence(plan_id: str, is_approved: bool):
    """Load a plan from persistence into session state."""
    try:
        persistence = get_persistence_service()
        plan_data = persistence.load_plan(plan_id, is_approved=is_approved)
        
        if plan_data:
            # Restore session state
            st.session_state.plan_data = {
                "campaign": plan_data.get("campaign", {}),
                "limits": plan_data.get("limits", {}),
                "all_creatives": plan_data.get("all_creatives", []),
                "recommended_list": plan_data.get("recommended_list", {}),
            }
            st.session_state.selected_creatives = plan_data.get("selected_creatives", {})
            
            # If approved, also add to test_plans
            if is_approved:
                st.session_state.test_plans[plan_id] = plan_data
            
            # Go to selection step
            st.session_state.planner_step = "select"
            st.success(f"✅ Loaded plan: {plan_data.get('campaign', {}).get('name', 'Unknown')}")
            st.rerun()
        else:
            st.error("Could not load plan.")
    except Exception as e:
        st.error(f"Error loading plan: {e}")


def render_video_upload():
    """Render video file upload step (Step 2 - after media plan)."""
    st.markdown("### Step 2: Upload Video Creatives")
    
    campaign = st.session_state.plan_data.get("campaign", {})
    limits = st.session_state.plan_data.get("limits", {})
    service = st.session_state.video_service
    
    # Show campaign info from media plan
    media_plan_count = len(service.media_plan) if service and service.media_plan else 0
    st.markdown(f"**Campaign:** {campaign.get('name', 'N/A')} | **Creatives in Media Plan:** {media_plan_count}")
    st.info(f"📹 Upload your video files (MP4). We'll analyze them for duplicates and match them to your media plan. **Limit: {limits.get('video_limit', 8)} videos can be tested.**")
    
    # Check if video ingestion is available
    if not VIDEO_INGESTION_AVAILABLE:
        st.warning("""
        ⚠️ **Video processing libraries not installed.** 
        
        To enable video upload and similarity detection, install:
        ```
        pip install opencv-python imagehash rapidfuzz pillow
        ```
        
        For now, you can skip video upload and proceed with media plan data only.
        """)
        
        # Option to skip video upload
        if st.button("Skip Video Upload → Use Media Plan Only", type="secondary"):
            st.session_state.plan_data["videos_processed"] = False
            st.session_state.planner_step = "select"
            st.rerun()
        
        # Navigation
        st.markdown("---")
        if st.button("← Back to Media Plan"):
            st.session_state.planner_step = "media_plan"
            st.rerun()
        return
    
    st.markdown("---")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload Video Files (MP4)",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=True,
        help="Upload all your video creatives. We'll detect duplicates and similar videos."
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} videos uploaded**")
        
        # Process videos button
        if st.button("Process Videos", type="primary"):
            with st.spinner("Processing videos... This may take a minute."):
                # Save uploaded files to temp directory
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                progress = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                    progress.progress((i + 1) / len(uploaded_files))
                
                # Ingest videos
                st.info("Extracting video metadata and computing similarity hashes...")
                try:
                    service.ingest_videos_batch(file_paths)
                    st.session_state.plan_data["videos_processed"] = True
                    st.session_state.plan_data["video_count"] = len(service.videos)
                    st.success(f"✅ Processed {len(service.videos)} videos!")
                except Exception as e:
                    st.error(f"Error processing videos: {e}")
                    return
        
        # Show processed videos
        if st.session_state.plan_data.get("videos_processed") and service.videos:
            st.markdown("#### Processed Videos")
            
            video_data = []
            for v in service.videos:
                video_data.append({
                    "Filename": v.filename,
                    "Duration": f"{v.duration_seconds:.1f}s",
                    "Resolution": f"{v.width}x{v.height}",
                    "Size (MB)": f"{v.file_size_mb:.1f}",
                })
            
            st.dataframe(video_data, use_container_width=True, hide_index=True)
            
            # Continue button - now matches videos to media plan
            if st.button("Match Videos to Media Plan →", type="primary", use_container_width=True):
                with st.spinner("Matching videos to media plan..."):
                    match_results = service.match_videos_to_media_plan()
                    st.session_state.plan_data["match_results"] = match_results
                    
                    # Detect similar videos
                    st.info("Detecting similar/duplicate videos...")
                    similarity_groups = service.detect_similar_videos()
                    st.session_state.plan_data["similarity_groups"] = similarity_groups
                    
                st.session_state.planner_step = "review_matches"
                st.rerun()
    
    # Option to skip video upload
    st.markdown("---")
    st.markdown("**Or skip video upload:**")
    if st.button("Skip → Use Media Plan Data Only", type="secondary"):
        st.session_state.plan_data["videos_processed"] = False
        st.session_state.plan_data["match_results"] = {"matched": [], "unmatched_videos": [], "unmatched_plan_entries": [], "match_rate": 0}
        st.session_state.plan_data["similarity_groups"] = []
        st.session_state.planner_step = "select"
        st.rerun()
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Media Plan"):
        st.session_state.planner_step = "media_plan"
        st.rerun()


def render_csv_fallback(limits):
    """Fallback to CSV upload when video processing isn't available."""
    st.markdown("---")
    st.markdown("### Alternative: Upload Creative List CSV")
    
    st.markdown("""
    Upload a CSV with your creative information:
    - `creative_name` - Name of the creative
    - `asset_type` - "video" or "display"
    - `channel` - Platform (YouTube, Facebook, etc.)
    - `impressions` - Planned impressions
    - `duration_seconds` - Video duration
    - `hypothesis` (optional) - What you're testing
    """)
    
    uploaded_file = st.file_uploader("Upload Creative List CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Rename columns
        column_map = {
            'creative_name': 'name',
            'creative_id': 'id',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        st.markdown("#### Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("Continue with CSV →", type="primary"):
            # Convert to creative list
            creatives = df.to_dict(orient="records")
            
            # Clean up
            for i, c in enumerate(creatives):
                if 'id' not in c or pd.isna(c.get('id')):
                    c['id'] = f"creative_{i+1}"
                if 'name' not in c or pd.isna(c.get('name')):
                    c['name'] = c['id']
                c['asset_type'] = str(c.get('asset_type', 'video')).lower()
                try:
                    c['impressions'] = int(float(c.get('impressions', 0)))
                except:
                    c['impressions'] = 0
            
            # Prioritize
            videos = [c for c in creatives if 'video' in c.get('asset_type', '')]
            videos_sorted = sorted(videos, key=lambda x: x.get('impressions', 0), reverse=True)
            
            for i, v in enumerate(videos_sorted):
                v['selected'] = i < limits.get('video_limit', 8)
                v['priority_rank'] = i + 1
            
            st.session_state.plan_data["all_creatives"] = videos_sorted
            st.session_state.selected_creatives = {c['id']: c.get('selected', False) for c in videos_sorted}
            st.session_state.planner_step = "select"
            st.rerun()


def render_media_plan_upload():
    """Render media plan upload as the FIRST step - extracts campaign info from the file."""
    st.markdown("### Step 1: Upload Media Plan")
    
    st.info("📊 Upload your media plan spreadsheet. We'll extract campaign details and creative information automatically.")
    
    st.markdown("""
    **Expected columns:**
    - `creative_name` or `asset_name` - Name of the creative
    - `channel` or `platform` - Where it will run (YouTube, Facebook, etc.)
    - `impressions` - Planned impressions
    - `duration_seconds` (optional) - Video length
    - `spend` or `budget` (optional) - Planned spend
    """)
    
    # Download sample
    if st.button("📥 Download Sample Media Plan"):
        sample_csv = create_sample_media_plan()
        st.download_button(
            "Download Sample CSV",
            data=sample_csv,
            file_name="sample_media_plan.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Media Plan",
        type=["csv", "xlsx", "xls"],
        help="Upload your media plan spreadsheet"
    )
    
    if uploaded_file:
        # Initialize video service if needed
        if st.session_state.video_service is None:
            if VIDEO_INGESTION_AVAILABLE:
                st.session_state.video_service = VideoIngestionService()
            else:
                st.error("Video ingestion service not available. Please install required packages.")
                return
        
        service = st.session_state.video_service
        
        with st.spinner("Parsing media plan..."):
            # Save to temp file
            suffix = ".xlsx" if "xls" in uploaded_file.name else ".csv"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            
            try:
                # Parse media plan
                service.parse_media_plan(temp_file.name)
                
                # Also read raw data to extract campaign info
                if suffix == ".csv":
                    df = pd.read_csv(temp_file.name)
                else:
                    df = pd.read_excel(temp_file.name, sheet_name=0)
                
                st.success(f"✅ Parsed {len(service.media_plan)} creatives from media plan!")
            except Exception as e:
                st.error(f"Error parsing media plan: {e}")
                return
            finally:
                os.unlink(temp_file.name)
        
        # Extract campaign info from media plan
        campaign_info = _extract_campaign_info(service.media_plan, df, uploaded_file.name)
        st.session_state.plan_data["campaign"] = campaign_info
        
        # Calculate limits based on total spend
        rules_engine = get_rules_engine()
        total_spend = sum(e.spend for e in service.media_plan if e.spend)
        total_impressions = sum(e.impressions for e in service.media_plan if e.impressions)
        
        # Estimate budget from spend (assume spend is ~1.5% of total budget)
        estimated_budget = max(total_spend * 66, 1000000) if total_spend > 0 else 10000000
        limits = rules_engine.get_limits_for_budget(estimated_budget)
        st.session_state.plan_data["limits"] = limits
        
        # Show campaign summary
        st.markdown("---")
        st.markdown("#### 📋 Campaign Summary (Extracted from Media Plan)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Campaign", campaign_info.get("name", "N/A"))
        with col2:
            st.metric("Total Creatives", len(service.media_plan))
        with col3:
            st.metric("Total Impressions", f"{total_impressions:,.0f}")
        with col4:
            st.metric("Total Spend", f"${total_spend:,.0f}")
        
        # Show testing limits
        st.markdown("#### 🎯 Testing Limits")
        limit_cols = st.columns(4)
        with limit_cols[0]:
            st.metric("Video Limit", limits["video_limit"])
        with limit_cols[1]:
            st.metric("Display Limit", limits["display_limit"])
        with limit_cols[2]:
            cost = rules_engine.calculate_cost(limits["video_limit"], limits["display_limit"])
            st.metric("Max Testing Cost", f"${cost:,.0f}")
        with limit_cols[3]:
            video_count = sum(1 for e in service.media_plan if e.duration_seconds and e.duration_seconds > 0)
            st.metric("Videos in Plan", video_count)
        
        # Show parsed entries
        st.markdown("---")
        st.markdown("#### Media Plan Entries")
        plan_data = []
        for entry in service.media_plan:
            plan_data.append({
                "Creative Name": entry.creative_name,
                "Channel": entry.channel or "N/A",
                "Impressions": f"{entry.impressions:,}" if entry.impressions else "N/A",
                "Spend": f"${entry.spend:,.0f}" if entry.spend else "N/A",
                "Duration": f"{entry.duration_seconds}s" if entry.duration_seconds else "N/A",
            })
        
        st.dataframe(plan_data, use_container_width=True, hide_index=True)
        
        # Continue button
        if st.button("Continue to Video Upload →", type="primary", use_container_width=True):
            st.session_state.planner_step = "videos"
            st.rerun()


def _extract_campaign_info(media_plan_entries, raw_df, filename):
    """Extract campaign information from media plan data."""
    import re
    from datetime import date, timedelta
    
    # Try to extract campaign name from filename
    # e.g., "Pixel_10_Q4_2025_media_plan.xlsx" -> "Pixel 10 Q4 2025"
    base_name = filename.replace("_media_plan", "").replace(".xlsx", "").replace(".csv", "").replace(".xls", "")
    campaign_name = base_name.replace("_", " ").title()
    
    # Try to extract brand from creative names
    brand = "Unknown"
    if media_plan_entries:
        first_creative = media_plan_entries[0].creative_name
        # Common brand patterns
        if "pixel" in first_creative.lower():
            brand = "Pixel"
        elif "search" in first_creative.lower():
            brand = "Google Search"
        elif "google" in first_creative.lower():
            brand = "Google"
        else:
            # Use first word of first creative
            brand = first_creative.split()[0] if first_creative else "Unknown"
    
    # Try to extract quarter and year from filename or creative names
    quarter = None
    year = None
    
    # Pattern: Q1, Q2, Q3, Q4
    q_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
    if q_match:
        quarter = f"Q{q_match.group(1)}"
    
    # Pattern: 2023, 2024, 2025, 2026
    y_match = re.search(r'(202[3-6])', filename)
    if y_match:
        year = int(y_match.group(1))
    
    # Default dates
    if year and quarter:
        quarter_months = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
        start_month = quarter_months.get(quarter, 1)
        start_date = date(year, start_month, 1)
        end_date = date(year, start_month + 2, 28)
    else:
        start_date = date.today() + timedelta(days=30)
        end_date = date.today() + timedelta(days=120)
    
    # Calculate total budget from spend (spend is typically ~1.5% of media budget)
    total_spend = sum(e.spend for e in media_plan_entries if e.spend)
    estimated_budget = max(total_spend * 66, 10000000) if total_spend > 0 else 10000000
    
    return {
        "id": str(uuid.uuid4())[:8],
        "name": campaign_name,
        "brand": {"id": brand.lower().replace(" ", "_"), "name": brand},
        "budget": estimated_budget,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "primary_kpi": "awareness",  # Valid KPIType enum value
        "quarter": quarter,
        "year": year,
    }


def render_match_review():
    """Render the matching and similarity review step."""
    st.markdown("### Step 3: Review Matches & Similar Videos")
    
    service = st.session_state.video_service
    match_results = st.session_state.plan_data.get("match_results", {})
    similarity_groups = st.session_state.plan_data.get("similarity_groups", [])
    
    # Match summary
    st.markdown("#### 🔗 Video-Media Plan Matching")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match Rate", f"{match_results.get('match_rate', 0):.0f}%")
    with col2:
        st.metric("Matched", len(match_results.get('matched', [])))
    with col3:
        unmatched = len(match_results.get('unmatched_videos', []))
        st.metric("Unmatched Videos", unmatched, delta_color="inverse" if unmatched > 0 else "off")
    
    # Show matches
    if match_results.get('matched'):
        with st.expander("✅ View Matches", expanded=False):
            for match in match_results['matched']:
                confidence_color = "🟢" if match['confidence'] >= 90 else "🟡" if match['confidence'] >= 80 else "🔴"
                st.markdown(f"{confidence_color} **{match['video']}** → {match['media_plan_entry']} ({match['confidence']:.0f}%, {match['match_type']})")
    
    # Show unmatched videos with manual matching option
    unmatched_videos = match_results.get('unmatched_videos', [])
    unmatched_plan_entries = match_results.get('unmatched_plan_entries', [])
    
    if unmatched_videos and unmatched_plan_entries:
        st.markdown("---")
        st.markdown("#### 🔧 Manual Matching")
        st.warning(f"**{len(unmatched_videos)} videos** couldn't be automatically matched. You can manually match them below.")
        
        # Initialize manual matches in session state
        if "manual_matches" not in st.session_state:
            st.session_state.manual_matches = {}
        
        # Show each unmatched video with a dropdown to select matching media plan entry
        for video in unmatched_videos:
            col1, col2 = st.columns([2, 2])
            
            with col1:
                st.markdown(f"**📹 {video}**")
            
            with col2:
                # Get current selection if any
                current_match = st.session_state.manual_matches.get(video, "-- Select Media Plan Entry --")
                
                options = ["-- Select Media Plan Entry --", "-- No Match (Skip) --"] + unmatched_plan_entries
                
                # If already matched, show that option
                if current_match not in options and current_match != "-- Select Media Plan Entry --":
                    options.append(current_match)
                
                selected = st.selectbox(
                    f"Match for {video[:30]}...",
                    options=options,
                    index=options.index(current_match) if current_match in options else 0,
                    key=f"manual_match_{video}",
                    label_visibility="collapsed"
                )
                
                if selected and selected not in ["-- Select Media Plan Entry --", "-- No Match (Skip) --"]:
                    st.session_state.manual_matches[video] = selected
                elif selected == "-- No Match (Skip) --":
                    st.session_state.manual_matches[video] = None
        
        # Apply manual matches button
        if st.button("Apply Manual Matches", type="secondary"):
            matches_applied = 0
            for video, entry_name in st.session_state.manual_matches.items():
                if entry_name and entry_name not in ["-- Select Media Plan Entry --", "-- No Match (Skip) --"]:
                    if service.manual_match(video, entry_name):
                        matches_applied += 1
                        # Update match results
                        match_results['matched'].append({
                            'video': video,
                            'media_plan_entry': entry_name,
                            'confidence': 100.0,
                            'match_type': 'manual'
                        })
                        # Remove from unmatched lists
                        if video in match_results['unmatched_videos']:
                            match_results['unmatched_videos'].remove(video)
                        if entry_name in match_results['unmatched_plan_entries']:
                            match_results['unmatched_plan_entries'].remove(entry_name)
            
            if matches_applied > 0:
                # Recalculate match rate
                match_results['match_rate'] = len(match_results['matched']) / len(service.media_plan) * 100 if service.media_plan else 0
                st.session_state.plan_data["match_results"] = match_results
                st.session_state.manual_matches = {}
                st.success(f"✅ Applied {matches_applied} manual match(es)!")
                st.rerun()
            else:
                st.info("No matches to apply. Select entries from the dropdowns above.")
    
    elif unmatched_videos:
        with st.expander("⚠️ Unmatched Videos", expanded=True):
            st.warning("These videos couldn't be matched to the media plan (no unmatched media plan entries available):")
            for video in unmatched_videos:
                st.markdown(f"- {video}")
    
    st.markdown("---")
    
    # Similarity detection results
    st.markdown("#### 🔍 Similar/Duplicate Video Detection")
    
    if similarity_groups:
        st.warning(f"Found **{len(similarity_groups)} groups** of similar videos. We recommend testing only one from each group.")
        
        for group in similarity_groups:
            with st.expander(f"Group {group.group_id + 1}: {group.similarity_type.replace('_', ' ').title()}", expanded=True):
                st.markdown(f"**Reason:** {group.reason}")
                st.markdown(f"**Recommended for testing:** ✅ {group.recommended_for_testing}")
                
                st.markdown("**Videos in this group:**")
                for video in group.videos:
                    if video.filename == group.recommended_for_testing:
                        st.markdown(f"- ✅ **{video.filename}** (recommended - highest impressions)")
                    else:
                        st.markdown(f"- ❌ {video.filename} (will be excluded)")
    else:
        st.success("✅ No duplicate or highly similar videos detected!")
    
    st.markdown("---")
    
    # Get recommended list
    if st.button("Continue to Selection →", type="primary", use_container_width=True):
        recommended = service.get_recommended_test_list()
        st.session_state.plan_data["recommended_list"] = recommended
        
        # Build creative list for selection
        creatives = []
        for i, rec in enumerate(recommended['recommended']):
            creatives.append({
                'id': f"vid_{i+1}",
                'name': rec['filename'],
                'asset_type': 'video',
                'channel': rec['channel'] or 'Unknown',
                'impressions': rec['impressions'],
                'from_similarity_group': rec['from_similarity_group'],
                'excluded_similar': rec['excluded_similar'],
                'selected': i < st.session_state.plan_data.get('limits', {}).get('video_limit', 8),
            })
        
        st.session_state.plan_data["all_creatives"] = creatives
        st.session_state.selected_creatives = {c['id']: c['selected'] for c in creatives}
        st.session_state.planner_step = "select"
        st.rerun()
    
    # Navigation
    if st.button("← Back to Video Upload"):
        st.session_state.planner_step = "videos"
        st.rerun()


def render_selection_and_approval():
    """Render final selection and approval step."""
    st.markdown("### Step 4: Select Videos for Testing")
    
    campaign = st.session_state.plan_data.get("campaign", {})
    all_creatives = st.session_state.plan_data.get("all_creatives", [])
    limits = st.session_state.plan_data.get("limits", {})
    recommended_list = st.session_state.plan_data.get("recommended_list", {})
    
    if not all_creatives:
        st.warning("No creatives found. Please go back and upload videos.")
        if st.button("← Back"):
            st.session_state.planner_step = "videos"
            st.rerun()
        return
    
    rules_engine = get_rules_engine()
    rules = rules_engine.rules
    video_limit = limits.get("video_limit", 8)
    
    st.markdown(f"**Campaign:** {campaign.get('name')} | **Budget:** ${campaign.get('budget', 0):,.0f}")
    
    # Show duplicate removal summary
    if recommended_list:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Uploaded", recommended_list.get('total_uploaded', 0))
        with col2:
            st.metric("After Deduplication", recommended_list.get('total_recommended', 0))
        with col3:
            st.metric("Duplicates Removed", recommended_list.get('duplicates_removed', 0))
    
    st.info(f"Select up to **{video_limit} videos** for testing. Videos are sorted by impressions.")
    
    st.markdown("---")
    
    # Video selection with checkboxes
    st.markdown("#### 🎬 Select Videos for Testing")
    
    selected_ids = []
    
    for creative in all_creatives:
        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1.5, 1.5, 2])
        
        with col1:
            current_selected = st.session_state.selected_creatives.get(creative['id'], creative.get('selected', False))
            
            # Count currently selected
            current_count = sum(1 for c in all_creatives if st.session_state.selected_creatives.get(c['id'], False))
            at_limit = current_count >= video_limit and not current_selected
            
            selected = st.checkbox(
                "Select",
                value=current_selected,
                key=f"sel_{creative['id']}",
                disabled=at_limit,
                label_visibility="collapsed"
            )
            st.session_state.selected_creatives[creative['id']] = selected
            
            if selected:
                selected_ids.append(creative['id'])
        
        with col2:
            name = creative.get('name', 'N/A')
            if selected:
                st.markdown(f"**✓ {name}**")
            else:
                st.markdown(name)
        
        with col3:
            imps = creative.get('impressions', 0)
            st.caption(f"{imps/1000000:.1f}M imps")
        
        with col4:
            channel = creative.get('channel', 'N/A')
            st.caption(channel)
        
        with col5:
            excluded = creative.get('excluded_similar', [])
            if excluded:
                st.caption(f"⚠️ Excludes {len(excluded)} similar")
    
    st.markdown("---")
    
    # Real-time cost calculation
    num_selected = len(selected_ids)
    cost = rules_engine.calculate_cost(num_selected, 0)
    testing_budget = campaign.get("budget", 0) * 0.015
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Videos Selected", f"{num_selected}/{video_limit}")
    with col2:
        st.metric("Total Cost", f"${cost:,.0f}")
    with col3:
        remaining = testing_budget - cost
        st.metric("Budget Remaining", f"${remaining:,.0f}")
    with col4:
        video_cost = rules.costs.video_cost
        st.metric("Cost per Video", f"${video_cost:,.0f}")
    
    st.markdown("---")
    
    # Navigation and approval
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Review"):
            st.session_state.planner_step = "review_matches"
            st.rerun()
    
    with col2:
        if st.button("Reset Selection"):
            for i, c in enumerate(all_creatives):
                st.session_state.selected_creatives[c['id']] = i < video_limit
            st.rerun()
    
    with col3:
        can_approve = num_selected > 0
        if st.button("✓ Approve Plan", type="primary", disabled=not can_approve, use_container_width=True):
            # Build final list
            final_creatives = [c for c in all_creatives if st.session_state.selected_creatives.get(c['id'], False)]
            
            # Build plan data
            plan_id = f"plan_{campaign['id']}_{date.today().strftime('%Y%m%d')}"
            plan_data = {
                "id": plan_id,
                "campaign": campaign,
                "creatives": final_creatives,
                "all_creatives": all_creatives,
                "status": "approved",
                "created_at": date.today().isoformat(),
                "cost": cost,
                "duplicates_removed": recommended_list.get('duplicates_removed', 0) if recommended_list else 0,
                "limits": limits,
                "selected_creatives": dict(st.session_state.selected_creatives),
            }
            
            # Save to session state
            st.session_state.test_plans[plan_id] = plan_data
            st.session_state.campaigns.append(campaign)
            
            # Save to disk if persistence is available
            if PERSISTENCE_AVAILABLE:
                try:
                    persistence = get_persistence_service()
                    saved_plan_id = persistence.save_plan(
                        campaign_id=campaign['id'],
                        plan_data=plan_data,
                        is_approved=True
                    )
                    st.session_state.plan_data['saved_plan_id'] = saved_plan_id
                except Exception as e:
                    st.warning(f"Plan approved but could not save to disk: {e}")
            
            # Show success
            st.balloons()
            st.success(f"""
            ### ✅ Plan Approved & Saved!
            
            **{campaign.get('name')}** is ready for testing.
            
            - **{num_selected} videos** selected for testing
            - **Total cost:** ${cost:,.0f}
            - **Duplicates removed:** {recommended_list.get('duplicates_removed', 0) if recommended_list else 0}
            
            {"✅ Plan saved to disk - will persist after restart." if PERSISTENCE_AVAILABLE else "⚠️ Plan saved to session only."}
            
            Go to the **Results** tab to upload test results when ready.
            """)
            
            # Reset for next plan
            if st.button("Create Another Plan"):
                st.session_state.planner_step = "media_plan"
                st.session_state.plan_data = {}
                st.session_state.selected_creatives = {}
                st.session_state.video_service = None
                st.rerun()
    
    if not can_approve:
        st.warning("Please select at least one video to approve the plan.")
    
    # Auto-save draft option
    st.markdown("---")
    if PERSISTENCE_AVAILABLE:
        if st.button("💾 Save Draft (without approving)", type="secondary"):
            draft_data = {
                "campaign": campaign,
                "all_creatives": all_creatives,
                "selected_creatives": dict(st.session_state.selected_creatives),
                "limits": limits,
                "recommended_list": recommended_list,
                "status": "draft",
            }
            try:
                persistence = get_persistence_service()
                draft_id = persistence.save_plan(
                    campaign_id=campaign['id'],
                    plan_data=draft_data,
                    is_approved=False
                )
                st.success(f"✅ Draft saved! You can load it later from the sidebar.")
            except Exception as e:
                st.error(f"Failed to save draft: {e}")



