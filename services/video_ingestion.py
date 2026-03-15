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
        
        # Normalize column names - handle spaces, parentheses, special chars
        df.columns = (df.columns
            .str.lower()
            .str.strip()
            .str.replace(' ', '_', regex=False)
            .str.replace('(', '', regex=False)
            .str.replace(')', '', regex=False)
            .str.replace('$', '', regex=False)
        )
        
        # Map common column variations - but only if target doesn't already exist
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
            
            # Channel - only map if 'channel' doesn't exist
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
            'spend_': 'spend',  # After removing $
            
            # Duration
            'length': 'duration_seconds',
            'seconds': 'duration_seconds',
            'duration': 'duration_seconds',
            'video_length': 'duration_seconds',
            'duration_s': 'duration_seconds',
        }
        
        # Only rename columns if the target column doesn't already exist
        cols_to_rename = {}
        for old_col, new_col in column_map.items():
            if old_col in df.columns and new_col not in df.columns:
                cols_to_rename[old_col] = new_col
        
        df = df.rename(columns=cols_to_rename)
        
        # Remove duplicate columns by keeping only the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Parse entries
        entries = []
        
        # Skip rows that are likely summary/total rows
        summary_keywords = ['summary', 'total', 'subtotal', 'grand total', 'count:', 'sum:']
        
        # Helper function to safely get value from row
        def safe_get(row, col, default=None):
            """Safely get a value from a row, handling missing columns and NaN."""
            if col not in row.index:
                return default
            val = row[col]
            # Handle case where val might be a Series (shouldn't happen but just in case)
            if hasattr(val, 'iloc'):
                val = val.iloc[0] if len(val) > 0 else default
            if pd.isna(val):
                return default
            return val
        
        for _, row in df.iterrows():
            name = safe_get(row, 'creative_name', '')
            name = str(name) if name else ''
            
            # Skip empty or NaN names
            if not name or name == 'nan' or name.lower() == 'nan':
                continue
            
            # Skip summary/header rows
            if any(keyword in name.lower() for keyword in summary_keywords):
                continue
            
            # Skip rows that look like labels (e.g., "Total Creatives:", "For Testing:")
            if name.endswith(':'):
                continue
            
            # Safely extract values
            creative_id = safe_get(row, 'creative_id')
            channel = safe_get(row, 'channel')
            impressions = safe_get(row, 'impressions', 0)
            spend = safe_get(row, 'spend', 0)
            duration = safe_get(row, 'duration_seconds', 0)
            
            entry = MediaPlanEntry(
                creative_name=name,
                creative_name_clean=self._normalize_name(name),
                creative_id=str(creative_id) if creative_id else None,
                channel=str(channel) if channel else None,
                impressions=int(float(impressions)) if impressions else 0,
                spend=float(spend) if spend else 0.0,
                duration_seconds=int(float(duration)) if duration else None,
            )
            entries.append(entry)
        
        self.media_plan = entries
        return entries
    
    def match_videos_to_media_plan(self) -> dict:
        """
        Match ingested videos to media plan entries using fuzzy matching.
        
        Uses multiple strategies:
        1. Exact match on normalized names
        2. Fuzzy match using token_sort_ratio
        3. Partial matching for YouTube-style filenames
        4. Bidirectional matching (video→plan and plan→video)
        
        Returns:
            Dict with match results and unmatched items
        """
        if not RAPIDFUZZ_AVAILABLE:
            raise ImportError("rapidfuzz or fuzzywuzzy required. Install with: pip install rapidfuzz")
        
        matched = []
        matched_videos = set()
        matched_entries = set()
        
        # Create lookups
        video_by_clean_name = {v.filename_clean: v for v in self.videos}
        entry_by_clean_name = {e.creative_name_clean: e for e in self.media_plan}
        
        # Also create variant lookups for videos
        video_variants = {}  # variant -> video
        for video in self.videos:
            variants = self._extract_title_variants(video.filename)
            for variant in variants:
                if variant not in video_variants:
                    video_variants[variant] = video
        
        # Strategy 1: Exact match on normalized names
        for entry in self.media_plan:
            if entry.creative_name_clean in video_by_clean_name:
                video = video_by_clean_name[entry.creative_name_clean]
                if video.filename not in matched_videos:
                    self._record_match(video, entry, 100.0, 'exact', matched, matched_videos, matched_entries)
        
        # Strategy 2: Match video variants to media plan
        for variant, video in video_variants.items():
            if video.filename in matched_videos:
                continue
            
            # Check if variant matches any media plan entry
            if variant in entry_by_clean_name:
                entry = entry_by_clean_name[variant]
                if entry.creative_name not in matched_entries:
                    self._record_match(video, entry, 95.0, 'variant_exact', matched, matched_videos, matched_entries)
                    continue
            
            # Fuzzy match variant to media plan entries
            unmatched_entry_names = [e.creative_name_clean for e in self.media_plan if e.creative_name not in matched_entries]
            if unmatched_entry_names:
                best = process.extractOne(variant, unmatched_entry_names, scorer=fuzz.token_sort_ratio)
                if best and best[1] >= self.NAME_MATCH_THRESHOLD:
                    entry = entry_by_clean_name[best[0]]
                    if entry.creative_name not in matched_entries:
                        self._record_match(video, entry, best[1], 'variant_fuzzy', matched, matched_videos, matched_entries)
        
        # Strategy 3: Fuzzy match remaining media plan entries to video variants
        for entry in self.media_plan:
            if entry.creative_name in matched_entries:
                continue
            
            # Get all unmatched video variants
            unmatched_variants = [v for v, vid in video_variants.items() if vid.filename not in matched_videos]
            if not unmatched_variants:
                break
            
            best = process.extractOne(entry.creative_name_clean, unmatched_variants, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= self.NAME_MATCH_THRESHOLD:
                video = video_variants[best[0]]
                if video.filename not in matched_videos:
                    self._record_match(video, entry, best[1], 'fuzzy', matched, matched_videos, matched_entries)
        
        # Strategy 4: Try partial matching with lower threshold for remaining
        lower_threshold = 60  # More lenient
        for entry in self.media_plan:
            if entry.creative_name in matched_entries:
                continue
            
            unmatched_variants = [v for v, vid in video_variants.items() if vid.filename not in matched_videos]
            if not unmatched_variants:
                break
            
            # Try token_set_ratio which is more lenient with extra words
            best = process.extractOne(entry.creative_name_clean, unmatched_variants, scorer=fuzz.token_set_ratio)
            if best and best[1] >= lower_threshold:
                video = video_variants[best[0]]
                if video.filename not in matched_videos:
                    self._record_match(video, entry, best[1], 'partial', matched, matched_videos, matched_entries)
        
        # Collect unmatched items
        unmatched_videos = [v for v in self.videos if v.filename not in matched_videos]
        unmatched_plan_entries = [e for e in self.media_plan if e.creative_name not in matched_entries]
        
        # Store for manual matching
        self._unmatched_videos = unmatched_videos
        self._unmatched_plan_entries = unmatched_plan_entries
        
        return {
            'matched': matched,
            'unmatched_videos': [v.filename for v in unmatched_videos],
            'unmatched_plan_entries': [e.creative_name for e in unmatched_plan_entries],
            'match_rate': len(matched) / len(self.media_plan) * 100 if self.media_plan else 0
        }
    
    def _record_match(self, video, entry, confidence, match_type, matched_list, matched_videos, matched_entries):
        """Record a match between video and media plan entry."""
        video.matched_to_media_plan = True
        video.media_plan_row = {
            'creative_name': entry.creative_name,
            'channel': entry.channel,
            'impressions': entry.impressions,
            'spend': entry.spend,
        }
        entry.matched_video = video.filename
        entry.match_confidence = confidence
        
        matched_list.append({
            'video': video.filename,
            'media_plan_entry': entry.creative_name,
            'confidence': confidence,
            'match_type': match_type
        })
        
        matched_videos.add(video.filename)
        matched_entries.add(entry.creative_name)
    
    def manual_match(self, video_filename: str, media_plan_entry_name: str) -> bool:
        """
        Manually match a video to a media plan entry.
        
        Args:
            video_filename: The filename of the video
            media_plan_entry_name: The creative name from the media plan
            
        Returns:
            True if match was successful
        """
        video = None
        entry = None
        
        for v in self.videos:
            if v.filename == video_filename:
                video = v
                break
        
        for e in self.media_plan:
            if e.creative_name == media_plan_entry_name:
                entry = e
                break
        
        if video and entry:
            video.matched_to_media_plan = True
            video.media_plan_row = {
                'creative_name': entry.creative_name,
                'channel': entry.channel,
                'impressions': entry.impressions,
                'spend': entry.spend,
            }
            entry.matched_video = video.filename
            entry.match_confidence = 100.0  # Manual match
            return True
        
        return False
    
    def get_unmatched_for_manual_matching(self) -> dict:
        """Get unmatched videos and media plan entries for manual matching UI."""
        return {
            'unmatched_videos': [v.filename for v in getattr(self, '_unmatched_videos', [])],
            'unmatched_plan_entries': [e.creative_name for e in getattr(self, '_unmatched_plan_entries', [])]
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
        
        # Handle YouTube-style filenames: "Title | Subtitle [VideoID]"
        # Extract the main title before | or first [
        if '|' in name:
            name = name.split('|')[0]
        if '[' in name:
            name = name.split('[')[0]
        
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
    
    def _extract_title_variants(self, name: str) -> list[str]:
        """Extract multiple title variants from a filename for better matching."""
        variants = []
        
        # Remove extension first
        name = os.path.splitext(name)[0]
        
        # Original (normalized)
        variants.append(self._normalize_name(name))
        
        # If has YouTube-style format, also try the subtitle part
        if '|' in name:
            parts = name.split('|')
            # Main title
            variants.append(self._normalize_name(parts[0]))
            # Subtitle (without video ID)
            if len(parts) > 1:
                subtitle = parts[1].split('[')[0].strip()
                if subtitle:
                    variants.append(self._normalize_name(subtitle))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique_variants.append(v)
        
        return unique_variants
    
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
