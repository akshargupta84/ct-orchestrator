"""
CT Planner Page - With Video Upload and Similarity Detection.

Features:
1. Upload actual video files (MP4)
2. Upload media plan spreadsheet
3. Auto-match videos to media plan
4. Detect similar/duplicate videos
5. Prioritize and select creatives for testing
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


def render_planner():
    """Render the CT Planner page."""
    st.markdown("## 📋 Creative Testing Planner")
    st.markdown("Upload videos and media plan to automatically generate an optimized test plan.")
    
    # Initialize planner state
    if "planner_step" not in st.session_state:
        st.session_state.planner_step = "campaign"
    if "plan_data" not in st.session_state:
        st.session_state.plan_data = {}
    if "selected_creatives" not in st.session_state:
        st.session_state.selected_creatives = {}
    if "video_service" not in st.session_state:
        st.session_state.video_service = None
    
    # Handle legacy step names from old code
    legacy_step_map = {
        "input": "campaign",
        "creatives": "videos",
        "review": "select",
        "approved": "campaign",  # Reset if approved
    }
    if st.session_state.planner_step in legacy_step_map:
        st.session_state.planner_step = legacy_step_map[st.session_state.planner_step]
    
    # Progress indicator - Updated steps
    steps = ["Campaign Info", "Upload Videos", "Upload Media Plan", "Review Matches", "Select & Approve"]
    step_keys = ["campaign", "videos", "media_plan", "review_matches", "select"]
    
    # Ensure current step is valid
    if st.session_state.planner_step not in step_keys:
        st.session_state.planner_step = "campaign"
    
    current_step_idx = step_keys.index(st.session_state.planner_step)
    
    cols = st.columns(5)
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
    if st.session_state.planner_step == "campaign":
        render_campaign_input()
    elif st.session_state.planner_step == "videos":
        render_video_upload()
    elif st.session_state.planner_step == "media_plan":
        render_media_plan_upload()
    elif st.session_state.planner_step == "review_matches":
        render_match_review()
    elif st.session_state.planner_step == "select":
        render_selection_and_approval()


def render_campaign_input():
    """Render campaign details input form - simplified."""
    st.markdown("### Step 1: Campaign Information")
    
    # Get existing data for persistence
    existing = st.session_state.plan_data.get("campaign", {})
    
    with st.form("campaign_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input(
                "Campaign Name",
                value=existing.get("name", ""),
                placeholder="e.g., Summer 2025 Brand Campaign"
            )
            
            brand_name = st.text_input(
                "Brand",
                value=existing.get("brand", {}).get("name", ""),
                placeholder="e.g., Brand X"
            )
            
            budget = st.number_input(
                "Campaign Budget ($)",
                min_value=100000,
                max_value=500000000,
                value=existing.get("budget", 10000000),
                step=1000000,
                format="%d"
            )
        
        with col2:
            # Parse existing dates or use defaults
            default_start = date.today() + timedelta(days=45)
            default_end = date.today() + timedelta(days=135)
            
            if existing.get("start_date"):
                try:
                    default_start = date.fromisoformat(existing["start_date"])
                except:
                    pass
            if existing.get("end_date"):
                try:
                    default_end = date.fromisoformat(existing["end_date"])
                except:
                    pass
            
            start_date = st.date_input("Campaign Start Date", value=default_start)
            end_date = st.date_input("Campaign End Date", value=default_end)
            
            # Get KPI options
            kpi_options = [k.value for k in KPIType]
            default_kpi_idx = 0
            if existing.get("primary_kpi") in kpi_options:
                default_kpi_idx = kpi_options.index(existing["primary_kpi"])
            
            primary_kpi = st.selectbox(
                "Primary KPI",
                options=kpi_options,
                index=default_kpi_idx,
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        # Show limits based on budget
        rules_engine = get_rules_engine()
        limits = rules_engine.get_limits_for_budget(budget)
        testing_budget = budget * 0.015
        
        st.markdown("---")
        st.markdown("#### Testing Limits for This Budget")
        limit_cols = st.columns(4)
        with limit_cols[0]:
            st.metric("Video Limit", limits["video_limit"])
        with limit_cols[1]:
            st.metric("Display Limit", limits["display_limit"])
        with limit_cols[2]:
            cost = rules_engine.calculate_cost(limits["video_limit"], limits["display_limit"])
            st.metric("Max Cost", f"${cost:,.0f}")
        with limit_cols[3]:
            st.metric("Testing Budget", f"${testing_budget:,.0f}")
        
        submitted = st.form_submit_button("Continue to Video Upload →", type="primary", use_container_width=True)
        
        if submitted:
            if not campaign_name:
                st.error("Please enter a campaign name")
            elif not brand_name:
                st.error("Please enter a brand name")
            else:
                st.session_state.plan_data["campaign"] = {
                    "id": existing.get("id", str(uuid.uuid4())[:8]),
                    "name": campaign_name,
                    "brand": {"id": brand_name.lower().replace(" ", "_"), "name": brand_name},
                    "budget": budget,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "primary_kpi": primary_kpi,
                }
                st.session_state.plan_data["limits"] = limits
                st.session_state.planner_step = "videos"
                st.rerun()


def render_video_upload():
    """Render video file upload step."""
    st.markdown("### Step 2: Upload Video Creatives")
    
    campaign = st.session_state.plan_data.get("campaign", {})
    limits = st.session_state.plan_data.get("limits", {})
    
    st.markdown(f"**Campaign:** {campaign.get('name', 'N/A')} | **Budget:** ${campaign.get('budget', 0):,.0f}")
    st.info(f"📹 Upload your video files (MP4). We'll analyze them for duplicates and match them to your media plan. **Limit: {limits.get('video_limit', 0)} videos can be tested.**")
    
    # Check if video ingestion is available
    if not VIDEO_INGESTION_AVAILABLE:
        st.warning("""
        ⚠️ **Video processing libraries not installed.** 
        
        To enable video upload and similarity detection, install:
        ```
        pip install opencv-python imagehash rapidfuzz pillow
        ```
        
        For now, you can use the **CSV upload** method instead.
        """)
        
        # Fallback to CSV method
        render_csv_fallback(limits)
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
        
        # Initialize video service
        if st.session_state.video_service is None:
            st.session_state.video_service = VideoIngestionService()
        
        service = st.session_state.video_service
        
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
            
            # Continue button
            if st.button("Continue to Media Plan Upload →", type="primary", use_container_width=True):
                st.session_state.planner_step = "media_plan"
                st.rerun()
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Campaign Info"):
        st.session_state.planner_step = "campaign"
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
    """Render media plan spreadsheet upload step."""
    st.markdown("### Step 3: Upload Media Plan")
    
    campaign = st.session_state.plan_data.get("campaign", {})
    video_count = st.session_state.plan_data.get("video_count", 0)
    
    st.markdown(f"**Campaign:** {campaign.get('name', 'N/A')} | **Videos Uploaded:** {video_count}")
    st.info("📊 Upload your media plan spreadsheet. We'll match it to your video files automatically.")
    
    st.markdown("""
    **Expected columns:**
    - `creative_name` or `asset_name` - Name of the creative
    - `channel` or `platform` - Where it will run (YouTube, Facebook, etc.)
    - `impressions` - Planned impressions
    - `duration_seconds` (optional) - Video length
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
        # Save and parse
        service = st.session_state.video_service
        
        if service is None:
            st.error("Video service not initialized. Please go back and process videos first.")
            return
        
        with st.spinner("Parsing media plan..."):
            # Save to temp file
            suffix = ".xlsx" if "xls" in uploaded_file.name else ".csv"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            
            try:
                service.parse_media_plan(temp_file.name)
                st.success(f"✅ Parsed {len(service.media_plan)} entries from media plan!")
            except Exception as e:
                st.error(f"Error parsing media plan: {e}")
                return
            finally:
                os.unlink(temp_file.name)
        
        # Show parsed entries
        st.markdown("#### Media Plan Entries")
        plan_data = []
        for entry in service.media_plan:
            plan_data.append({
                "Creative Name": entry.creative_name,
                "Channel": entry.channel or "N/A",
                "Impressions": f"{entry.impressions:,}" if entry.impressions else "N/A",
                "Duration": f"{entry.duration_seconds}s" if entry.duration_seconds else "N/A",
            })
        
        st.dataframe(plan_data, use_container_width=True, hide_index=True)
        
        # Continue button
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
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Video Upload"):
        st.session_state.planner_step = "videos"
        st.rerun()


def render_match_review():
    """Render the matching and similarity review step."""
    st.markdown("### Step 4: Review Matches & Similar Videos")
    
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
        with st.expander("View Matches", expanded=True):
            for match in match_results['matched']:
                confidence_color = "🟢" if match['confidence'] >= 90 else "🟡" if match['confidence'] >= 80 else "🔴"
                st.markdown(f"{confidence_color} **{match['video']}** → {match['media_plan_entry']} ({match['confidence']:.0f}% confidence, {match['match_type']})")
    
    # Show unmatched
    if match_results.get('unmatched_videos'):
        with st.expander("⚠️ Unmatched Videos", expanded=True):
            st.warning("These videos couldn't be matched to the media plan:")
            for video in match_results['unmatched_videos']:
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
    if st.button("← Back to Media Plan"):
        st.session_state.planner_step = "media_plan"
        st.rerun()


def render_selection_and_approval():
    """Render final selection and approval step."""
    st.markdown("### Step 5: Select Videos for Testing")
    
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
            
            # Save plan
            plan_id = f"plan_{campaign['id']}_{date.today().strftime('%Y%m%d')}"
            st.session_state.test_plans[plan_id] = {
                "id": plan_id,
                "campaign": campaign,
                "creatives": final_creatives,
                "all_creatives": all_creatives,
                "status": "approved",
                "created_at": date.today().isoformat(),
                "cost": cost,
                "duplicates_removed": recommended_list.get('duplicates_removed', 0) if recommended_list else 0,
            }
            st.session_state.campaigns.append(campaign)
            
            # Show success
            st.balloons()
            st.success(f"""
            ### ✅ Plan Approved!
            
            **{campaign.get('name')}** is ready for testing.
            
            - **{num_selected} videos** selected for testing
            - **Total cost:** ${cost:,.0f}
            - **Duplicates removed:** {recommended_list.get('duplicates_removed', 0) if recommended_list else 0}
            
            Go to the **Results** tab to upload test results when ready.
            """)
            
            # Reset for next plan
            if st.button("Create Another Plan"):
                st.session_state.planner_step = "campaign"
                st.session_state.plan_data = {}
                st.session_state.selected_creatives = {}
                st.session_state.video_service = None
                st.rerun()
    
    if not can_approve:
        st.warning("Please select at least one video to approve the plan.")



