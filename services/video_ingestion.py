"""
Video Ingestion Service.

Handles:
1. Video file upload and metadata extraction
2. Media plan parsing and matching
3. Video similarity detection to identify duplicates/near-duplicates
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import timedelta
import re

# Video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Image hashing for similarity
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

# Fuzzy string matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        RAPIDFUZZ_AVAILABLE = True
    except ImportError:
        RAPIDFUZZ_AVAILABLE = False

import pandas as pd


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    file_path: str
    filename: str
    filename_clean: str  # Normalized for matching
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    file_size_mb: float
    
    # Hashes for similarity detection
    first_frame_hash: Optional[str] = None
    last_frame_hash: Optional[str] = None
    middle_frame_hash: Optional[str] = None
    
    # Thumbnails (base64 encoded)
    thumbnail_first: Optional[str] = None
    thumbnail_last: Optional[str] = None
    
    # Matching
    matched_to_media_plan: bool = False
    media_plan_row: Optional[dict] = None
    
    # Similarity
    similarity_group: Optional[int] = None
    similar_videos: list = field(default_factory=list)


@dataclass 
class MediaPlanEntry:
    """Entry from the media plan spreadsheet."""
    creative_name: str
    creative_name_clean: str  # Normalized for matching
    creative_id: Optional[str] = None
    channel: Optional[str] = None
    platform: Optional[str] = None
    impressions: int = 0
    spend: float = 0.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_seconds: Optional[int] = None
    format: Optional[str] = None
    
    # Matching
    matched_video: Optional[str] = None
    match_confidence: float = 0.0


@dataclass
class SimilarityGroup:
    """Group of similar videos."""
    group_id: int
    videos: list  # List of VideoMetadata
    similarity_type: str  # "exact_duplicate", "near_duplicate", "similar_content"
    reason: str  # Why they were grouped
    recommended_for_testing: Optional[str] = None  # Which one to test


class VideoIngestionService:
    """
    Service for ingesting video files and matching to media plans.
    """
    
    # Similarity thresholds
    HASH_SIMILARITY_THRESHOLD = 10  # Hamming distance for perceptual hashes
    DURATION_TOLERANCE_SECONDS = 0.5  # How close durations must be
    NAME_MATCH_THRESHOLD = 80  # Fuzzy match score threshold
    
    def __init__(self, upload_dir: str = None):
        """
        Initialize the video ingestion service.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = upload_dir or tempfile.mkdtemp()
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        
        self.videos: list[VideoMetadata] = []
        self.media_plan: list[MediaPlanEntry] = []
        self.similarity_groups: list[SimilarityGroup] = []
    
    def ingest_video(self, file_path: str) -> VideoMetadata:
        """
        Ingest a single video file and extract metadata.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            VideoMetadata object
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for video processing. Install with: pip install opencv-python")
        
        # Open video
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        # Extract basic metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        metadata = VideoMetadata(
            file_path=file_path,
            filename=filename,
            filename_clean=self._normalize_name(filename),
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            file_size_mb=file_size,
        )
        
        # Extract frame hashes for similarity detection
        if IMAGEHASH_AVAILABLE:
            metadata.first_frame_hash = self._get_frame_hash(cap, 0)
            metadata.middle_frame_hash = self._get_frame_hash(cap, frame_count // 2)
            metadata.last_frame_hash = self._get_frame_hash(cap, frame_count - 10)  # Near end
        
        cap.release()
        
        self.videos.append(metadata)
        return metadata
    
    def ingest_videos_batch(self, file_paths: list[str]) -> list[VideoMetadata]:
        """Ingest multiple videos."""
        results = []
        for path in file_paths:
            try:
                metadata = self.ingest_video(path)
                results.append(metadata)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return results
    
    def parse_media_plan(self, file_path: str, sheet_name: str = None) -> list[MediaPlanEntry]:
        """
        Parse a media plan Excel/CSV file.
        
        Expected columns (flexible naming):
        - creative_name / creative / name / asset_name
        - creative_id / id / asset_id
        - channel / platform / placement
        - impressions / imps / planned_impressions
        - spend / budget / cost
        - duration / length / seconds
        
        Args:
            file_path: Path to Excel or CSV file
            sheet_name: Sheet name for Excel files
            
        Returns:
            List of MediaPlanEntry objects
        """
        # Read file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # For Excel, read the first sheet if no sheet_name specified
            if sheet_name is None:
                df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Map common column variations
        column_map = {
            # Creative name
            'creative': 'creative_name',
            'name': 'creative_name',
            'asset_name': 'creative_name',
            'asset': 'creative_name',
            'video_name': 'creative_name',
            
            # Creative ID
            'id': 'creative_id',
            'asset_id': 'creative_id',
            'video_id': 'creative_id',
            
            # Channel
            'platform': 'channel',
            'placement': 'channel',
            'media_channel': 'channel',
            
            # Impressions
            'imps': 'impressions',
            'planned_impressions': 'impressions',
            'planned_imps': 'impressions',
            'total_impressions': 'impressions',
            
            # Spend
            'budget': 'spend',
            'cost': 'spend',
            'planned_spend': 'spend',
            'spend_$': 'spend',
            
            # Duration
            'length': 'duration_seconds',
            'seconds': 'duration_seconds',
            'duration': 'duration_seconds',
            'video_length': 'duration_seconds',
            'duration_s': 'duration_seconds',
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Parse entries
        entries = []
        for _, row in df.iterrows():
            name = str(row.get('creative_name', ''))
            if not name or name == 'nan':
                continue
            
            entry = MediaPlanEntry(
                creative_name=name,
                creative_name_clean=self._normalize_name(name),
                creative_id=str(row.get('creative_id', '')) if pd.notna(row.get('creative_id')) else None,
                channel=str(row.get('channel', '')) if pd.notna(row.get('channel')) else None,
                impressions=int(row.get('impressions', 0)) if pd.notna(row.get('impressions')) else 0,
                spend=float(row.get('spend', 0)) if pd.notna(row.get('spend')) else 0.0,
                duration_seconds=int(row.get('duration_seconds', 0)) if pd.notna(row.get('duration_seconds')) else None,
            )
            entries.append(entry)
        
        self.media_plan = entries
        return entries
    
    def match_videos_to_media_plan(self) -> dict:
        """
        Match ingested videos to media plan entries using fuzzy matching.
        
        Returns:
            Dict with match results and unmatched items
        """
        if not RAPIDFUZZ_AVAILABLE:
            raise ImportError("rapidfuzz or fuzzywuzzy required. Install with: pip install rapidfuzz")
        
        matched = []
        unmatched_videos = []
        unmatched_plan_entries = []
        
        # Create lookup for video names
        video_names = {v.filename_clean: v for v in self.videos}
        
        for entry in self.media_plan:
            # Try exact match first
            if entry.creative_name_clean in video_names:
                video = video_names[entry.creative_name_clean]
                video.matched_to_media_plan = True
                video.media_plan_row = {
                    'creative_name': entry.creative_name,
                    'channel': entry.channel,
                    'impressions': entry.impressions,
                    'spend': entry.spend,
                }
                entry.matched_video = video.filename
                entry.match_confidence = 100.0
                matched.append({
                    'video': video.filename,
                    'media_plan_entry': entry.creative_name,
                    'confidence': 100.0,
                    'match_type': 'exact'
                })
                continue
            
            # Try fuzzy match
            best_match = process.extractOne(
                entry.creative_name_clean,
                list(video_names.keys()),
                scorer=fuzz.token_sort_ratio
            )
            
            if best_match and best_match[1] >= self.NAME_MATCH_THRESHOLD:
                video = video_names[best_match[0]]
                video.matched_to_media_plan = True
                video.media_plan_row = {
                    'creative_name': entry.creative_name,
                    'channel': entry.channel,
                    'impressions': entry.impressions,
                    'spend': entry.spend,
                }
                entry.matched_video = video.filename
                entry.match_confidence = best_match[1]
                matched.append({
                    'video': video.filename,
                    'media_plan_entry': entry.creative_name,
                    'confidence': best_match[1],
                    'match_type': 'fuzzy'
                })
            else:
                unmatched_plan_entries.append(entry)
        
        # Find unmatched videos
        for video in self.videos:
            if not video.matched_to_media_plan:
                unmatched_videos.append(video)
        
        return {
            'matched': matched,
            'unmatched_videos': [v.filename for v in unmatched_videos],
            'unmatched_plan_entries': [e.creative_name for e in unmatched_plan_entries],
            'match_rate': len(matched) / len(self.media_plan) * 100 if self.media_plan else 0
        }
    
    def detect_similar_videos(self) -> list[SimilarityGroup]:
        """
        Detect similar or duplicate videos based on:
        1. Perceptual hash similarity (visual content)
        2. Duration matching
        3. Name similarity
        
        Returns:
            List of SimilarityGroup objects
        """
        groups = []
        group_id = 0
        processed = set()
        
        for i, video1 in enumerate(self.videos):
            if video1.filename in processed:
                continue
            
            similar = []
            
            for j, video2 in enumerate(self.videos):
                if i >= j or video2.filename in processed:
                    continue
                
                similarity_reasons = []
                
                # Check duration similarity
                duration_diff = abs(video1.duration_seconds - video2.duration_seconds)
                same_duration = duration_diff <= self.DURATION_TOLERANCE_SECONDS
                
                if same_duration:
                    similarity_reasons.append(f"Same duration ({video1.duration_seconds:.1f}s)")
                
                # Check perceptual hash similarity
                if IMAGEHASH_AVAILABLE and video1.first_frame_hash and video2.first_frame_hash:
                    first_frame_similar = self._hash_distance(
                        video1.first_frame_hash, video2.first_frame_hash
                    ) <= self.HASH_SIMILARITY_THRESHOLD
                    
                    last_frame_similar = self._hash_distance(
                        video1.last_frame_hash, video2.last_frame_hash
                    ) <= self.HASH_SIMILARITY_THRESHOLD
                    
                    middle_frame_similar = self._hash_distance(
                        video1.middle_frame_hash, video2.middle_frame_hash
                    ) <= self.HASH_SIMILARITY_THRESHOLD
                    
                    if first_frame_similar and middle_frame_similar and last_frame_similar:
                        similarity_reasons.append("All frames visually identical")
                    elif first_frame_similar and middle_frame_similar and not last_frame_similar:
                        similarity_reasons.append("Same content, different end card")
                    elif first_frame_similar and last_frame_similar:
                        similarity_reasons.append("Same opening and ending")
                
                # Check name similarity
                if RAPIDFUZZ_AVAILABLE:
                    name_score = fuzz.token_sort_ratio(
                        video1.filename_clean, video2.filename_clean
                    )
                    if name_score >= 80:
                        similarity_reasons.append(f"Similar names ({name_score}% match)")
                
                # If similar enough, add to group
                if len(similarity_reasons) >= 2 or "All frames visually identical" in str(similarity_reasons):
                    similar.append({
                        'video': video2,
                        'reasons': similarity_reasons
                    })
            
            if similar:
                # Determine similarity type
                all_reasons = [r for s in similar for r in s['reasons']]
                
                if "All frames visually identical" in str(all_reasons):
                    sim_type = "exact_duplicate"
                elif "different end card" in str(all_reasons):
                    sim_type = "near_duplicate"
                else:
                    sim_type = "similar_content"
                
                # Create group
                group_videos = [video1] + [s['video'] for s in similar]
                
                # Recommend which to test (highest impressions)
                recommended = max(
                    group_videos,
                    key=lambda v: v.media_plan_row.get('impressions', 0) if v.media_plan_row else 0
                )
                
                group = SimilarityGroup(
                    group_id=group_id,
                    videos=group_videos,
                    similarity_type=sim_type,
                    reason="; ".join(set(all_reasons)),
                    recommended_for_testing=recommended.filename
                )
                groups.append(group)
                
                # Mark as processed
                processed.add(video1.filename)
                for s in similar:
                    processed.add(s['video'].filename)
                    s['video'].similarity_group = group_id
                
                video1.similarity_group = group_id
                group_id += 1
        
        self.similarity_groups = groups
        return groups
    
    def get_recommended_test_list(self) -> list[dict]:
        """
        Get the recommended list of videos to test, excluding duplicates.
        
        Returns:
            List of video dicts with metadata and media plan info
        """
        recommended = []
        excluded_duplicates = []
        
        # Get videos from similarity groups (only recommended ones)
        grouped_videos = set()
        for group in self.similarity_groups:
            grouped_videos.update(v.filename for v in group.videos)
            
            # Add only the recommended video from each group
            rec_video = next(
                (v for v in group.videos if v.filename == group.recommended_for_testing),
                group.videos[0]
            )
            
            recommended.append({
                'filename': rec_video.filename,
                'duration': rec_video.duration_seconds,
                'channel': rec_video.media_plan_row.get('channel') if rec_video.media_plan_row else None,
                'impressions': rec_video.media_plan_row.get('impressions', 0) if rec_video.media_plan_row else 0,
                'from_similarity_group': True,
                'group_id': group.group_id,
                'excluded_similar': [v.filename for v in group.videos if v.filename != rec_video.filename],
            })
            
            # Track excluded
            for v in group.videos:
                if v.filename != rec_video.filename:
                    excluded_duplicates.append({
                        'filename': v.filename,
                        'reason': group.reason,
                        'recommended_instead': rec_video.filename,
                    })
        
        # Add ungrouped videos
        for video in self.videos:
            if video.filename not in grouped_videos:
                recommended.append({
                    'filename': video.filename,
                    'duration': video.duration_seconds,
                    'channel': video.media_plan_row.get('channel') if video.media_plan_row else None,
                    'impressions': video.media_plan_row.get('impressions', 0) if video.media_plan_row else 0,
                    'from_similarity_group': False,
                    'group_id': None,
                    'excluded_similar': [],
                })
        
        # Sort by impressions descending
        recommended.sort(key=lambda x: x['impressions'], reverse=True)
        
        return {
            'recommended': recommended,
            'excluded_duplicates': excluded_duplicates,
            'total_uploaded': len(self.videos),
            'total_recommended': len(recommended),
            'duplicates_removed': len(excluded_duplicates),
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a filename/creative name for matching."""
        # Remove extension
        name = os.path.splitext(name)[0]
        # Lowercase
        name = name.lower()
        # Remove common suffixes
        name = re.sub(r'_v\d+$', '', name)
        name = re.sub(r'_final$', '', name)
        name = re.sub(r'_rev\d*$', '', name)
        # Replace separators with spaces
        name = re.sub(r'[_\-\.]+', ' ', name)
        # Remove extra whitespace
        name = ' '.join(name.split())
        return name
    
    def _get_frame_hash(self, cap, frame_num: int) -> Optional[str]:
        """Get perceptual hash of a specific frame."""
        if not IMAGEHASH_AVAILABLE:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Compute perceptual hash
        phash = imagehash.phash(pil_image)
        return str(phash)
    
    def _hash_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes."""
        if not hash1 or not hash2:
            return 999  # Max distance if hashes missing
        
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2
        except:
            return 999


def create_sample_media_plan() -> str:
    """Create a sample media plan CSV for testing."""
    data = """creative_name,creative_id,channel,impressions,spend,duration_seconds
Summer_Hero_30s,VID001,YouTube,25000000,150000,30
Summer_Hero_15s,VID002,Facebook,40000000,80000,15
Product_Demo_30s,VID003,Display,30000000,100000,30
Testimonial_30s,VID004,Instagram,20000000,60000,30
Lifestyle_15s,VID005,TikTok,35000000,70000,15
Tutorial_60s,VID006,YouTube,15000000,120000,60
Summer_Hero_30s_v2,VID007,Connected TV,10000000,90000,30
Product_Demo_15s,VID008,Facebook,45000000,85000,15
"""
    return data
