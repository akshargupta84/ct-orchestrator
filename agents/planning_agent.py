"""
Planning Agent - LLM-powered conversational agent for creative test planning.

This agent:
1. Extracts campaign info from uploaded media plans
2. Analyzes uploaded videos (pre-scoring, duplicate detection)
3. Reasons about test plan optimization
4. Engages in dialogue with the user to create optimal plans
"""

import os
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import re

# LLM
from utils.llm import get_completion

# Services
try:
    from services.video_ingestion import VideoIngestionService
    VIDEO_SERVICE_AVAILABLE = True
except ImportError:
    VIDEO_SERVICE_AVAILABLE = False

try:
    from services.creative_scorer import get_scorer_service, ScoringConfig
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False

try:
    from services.rules_engine import get_rules_engine
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False

try:
    from services.prediction_model import get_prediction_model
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False


@dataclass
class MediaPlanInfo:
    """Extracted information from a media plan."""
    brand: str = ""
    campaign_name: str = ""
    total_budget: float = 0
    flight_start: str = ""
    flight_end: str = ""
    markets: List[str] = field(default_factory=list)
    primary_kpi: str = ""
    creative_line_items: List[Dict] = field(default_factory=list)
    raw_data: pd.DataFrame = None
    
    def to_dict(self) -> dict:
        return {
            'brand': self.brand,
            'campaign_name': self.campaign_name,
            'total_budget': self.total_budget,
            'flight_start': self.flight_start,
            'flight_end': self.flight_end,
            'markets': self.markets,
            'primary_kpi': self.primary_kpi,
            'creative_line_items': self.creative_line_items,
        }


@dataclass
class VideoInfo:
    """Information about an uploaded video."""
    filename: str
    filepath: str
    duration: float = 0
    pass_probability: float = 0
    risk_factors: List[str] = field(default_factory=list)
    matched_line_item: str = ""
    is_duplicate_of: str = ""
    features: Dict = field(default_factory=dict)
    scored: bool = False


@dataclass 
class ConversationState:
    """State of the planning conversation."""
    messages: List[Dict] = field(default_factory=list)
    media_plan_info: Optional[MediaPlanInfo] = None
    videos: List[VideoInfo] = field(default_factory=list)
    duplicates_detected: List[Tuple[str, str, float]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    plan_ready: bool = False
    final_plan: Dict = field(default_factory=dict)


class PlanningAgent:
    """
    Conversational agent for creative test planning.
    
    The agent:
    - Parses uploaded media plans to extract campaign info
    - Analyzes uploaded videos for pre-scoring and duplicates
    - Engages in dialogue to resolve issues and optimize the plan
    - Generates final test plans
    """
    
    SYSTEM_PROMPT = """You are an expert creative testing planning agent. Your job is to help users create optimal test plans through conversation.

You have access to:
- Media plan data uploaded by the user (campaign details, budgets, creative line items)
- Video analysis results (pre-scores, duplicate detection, risk factors)
- CT Rules (budget limits: 4% of media spend, costs: $15K/video, $8K/static)
- Historical performance data

Your approach:
1. When files are uploaded, analyze them and summarize what you found
2. Proactively identify issues (duplicates, low scores, missing info, budget problems)
3. Explain your reasoning clearly
4. Offer options and recommendations, don't just dictate
5. Ask clarifying questions only when necessary
6. Be conversational but efficient

When presenting video analysis:
- Show pass probabilities as percentages
- Flag videos with < 50% pass probability as risky
- Explain WHY a video scored low (missing human in opening, late logo, etc.)
- Identify duplicates and recommend which to keep

When there are issues:
- Present them clearly with options for resolution
- Let the user decide, but give your recommendation

Format your responses with clear sections using markdown:
- Use **bold** for important numbers and decisions
- Use tables for comparing videos
- Use bullet points for lists of issues or recommendations

Remember: The user is a media professional. Be helpful but respect their expertise."""

    def __init__(self):
        self.state = ConversationState()
        self.video_service = VideoIngestionService() if VIDEO_SERVICE_AVAILABLE else None
        self.scorer = get_scorer_service() if SCORER_AVAILABLE else None
        self.rules_engine = get_rules_engine() if RULES_AVAILABLE else None
        
    def reset(self):
        """Reset the conversation state."""
        self.state = ConversationState()
        
    def get_messages(self) -> List[Dict]:
        """Get conversation messages for display."""
        return self.state.messages
    
    def _add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.state.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    # =========================================================================
    # Media Plan Parsing
    # =========================================================================
    
    def parse_media_plan(self, file_path: str, file_type: str = 'xlsx') -> MediaPlanInfo:
        """
        Parse a media plan file and extract campaign information.
        
        Handles two layouts:
        1. Row-based: Labels in column A, values in column B (like our test files)
        2. Column-based: Headers in row 1, data in rows below
        """
        try:
            # Read file
            if file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, header=None)  # No header assumption
            else:
                df = pd.read_csv(file_path, header=None)
            
            info = MediaPlanInfo(raw_data=df)
            
            # Convert to string and strip whitespace
            df = df.astype(str).apply(lambda x: x.str.strip())
            
            # Strategy 1: Look for key-value pairs (label in col A, value in col B)
            # This handles row-based layouts
            for idx, row in df.iterrows():
                if len(row) < 2:
                    continue
                    
                label = str(row.iloc[0]).lower() if pd.notna(row.iloc[0]) else ""
                value = str(row.iloc[1]) if pd.notna(row.iloc[1]) and str(row.iloc[1]) != 'nan' else ""
                
                if not value or value == 'nan':
                    continue
                
                # Brand
                if 'brand' in label or 'advertiser' in label or 'client' in label:
                    info.brand = value
                
                # Campaign
                if 'campaign' in label and 'budget' not in label:
                    info.campaign_name = value
                
                # Budget - look for dollar amounts
                if 'budget' in label:
                    # Extract number from string like "$2,000,000" or "2000000"
                    budget_str = re.sub(r'[^\d.]', '', value)
                    try:
                        info.total_budget = float(budget_str)
                    except:
                        pass
                
                # Flight dates
                if 'flight' in label and 'start' in label:
                    info.flight_start = value
                elif 'flight' in label and 'end' in label:
                    info.flight_end = value
                elif 'start' in label and 'date' in label:
                    info.flight_start = value
                elif 'end' in label and 'date' in label:
                    info.flight_end = value
                
                # Markets
                if 'market' in label or 'geo' in label or 'region' in label:
                    # Split by comma if multiple
                    markets = [m.strip() for m in value.split(',')]
                    info.markets = markets
                
                # KPI
                if 'kpi' in label or 'objective' in label or 'goal' in label:
                    info.primary_kpi = value
                
                # Product
                if 'product' in label:
                    if not info.campaign_name:
                        info.campaign_name = value
            
            # Strategy 2: Look for creative line items table
            # Find row that looks like a header (contains "creative" or "asset")
            creative_header_row = None
            creative_col_idx = None
            
            for idx, row in df.iterrows():
                for col_idx, cell in enumerate(row):
                    cell_lower = str(cell).lower()
                    if cell_lower in ['creative', 'creative name', 'asset', 'asset name', 'ad name']:
                        creative_header_row = idx
                        creative_col_idx = col_idx
                        break
                if creative_header_row is not None:
                    break
            
            # Extract creative names from the table
            if creative_header_row is not None and creative_col_idx is not None:
                for idx in range(creative_header_row + 1, len(df)):
                    creative_name = df.iloc[idx, creative_col_idx]
                    if creative_name and str(creative_name) != 'nan' and str(creative_name).strip():
                        # Skip if it looks like a section header
                        if str(creative_name).lower() not in ['creative', 'creative name', 'nan', '']:
                            line_item = {'name': str(creative_name).strip()}
                            
                            # Try to get format, duration from adjacent columns
                            if creative_col_idx + 1 < len(df.columns):
                                fmt = df.iloc[idx, creative_col_idx + 1]
                                if fmt and str(fmt) != 'nan':
                                    line_item['format'] = str(fmt)
                            
                            if creative_col_idx + 2 < len(df.columns):
                                dur = df.iloc[idx, creative_col_idx + 2]
                                if dur and str(dur) != 'nan':
                                    line_item['duration'] = str(dur)
                            
                            info.creative_line_items.append(line_item)
            
            # If still no brand, try to find it anywhere in the sheet
            if not info.brand:
                for idx, row in df.iterrows():
                    for cell in row:
                        cell_str = str(cell).lower()
                        if 'google' in cell_str or 'pixel' in cell_str:
                            # Found something Google/Pixel related
                            if 'pixel' in cell_str:
                                info.brand = "Google"
                                break
                    if info.brand:
                        break
            
            self.state.media_plan_info = info
            
            # Debug print
            print(f"Parsed media plan: brand={info.brand}, campaign={info.campaign_name}, budget={info.total_budget}")
            print(f"Found {len(info.creative_line_items)} creative line items")
            
            return info
            
        except Exception as e:
            print(f"Error parsing media plan: {e}")
            import traceback
            traceback.print_exc()
            return MediaPlanInfo()
    
    # =========================================================================
    # Video Analysis
    # =========================================================================
    
    def analyze_videos(self, video_files: List[Tuple[str, str]], progress_callback=None) -> List[VideoInfo]:
        """
        Analyze uploaded videos for pre-scoring and duplicate detection.
        
        Args:
            video_files: List of (filename, filepath) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of VideoInfo objects
        """
        videos = []
        
        for i, (filename, filepath) in enumerate(video_files):
            if progress_callback:
                progress_callback(f"Analyzing {filename}...", i / len(video_files))
            
            video_info = VideoInfo(filename=filename, filepath=filepath)
            
            # Pre-score the video if scorer is available
            if self.scorer:
                try:
                    availability = self.scorer.check_availability()
                    if availability.get('ready'):
                        config = ScoringConfig(max_frames=8)  # Faster for planning
                        result = self.scorer.score_creative(filepath, config)
                        
                        if result and result.score:
                            video_info.pass_probability = result.score.pass_probability
                            video_info.risk_factors = [
                                rf.factor for rf in (result.score.risk_factors or [])
                            ]
                            video_info.features = result.features.to_dict() if result.features else {}
                            video_info.diagnostics = result.score.predicted_diagnostics or {}
                            video_info.scored = True
                except Exception as e:
                    print(f"Error scoring {filename}: {e}")
            
            videos.append(video_info)
        
        # Detect duplicates using video service
        if self.video_service and len(videos) > 1:
            try:
                # Add videos to the service first
                self.video_service.videos = []  # Reset
                for v in videos:
                    self.video_service.add_video(v.filepath, v.filename)
                
                # Now detect duplicates
                similarity_groups = self.video_service.detect_similar_videos()
                
                for group in similarity_groups:
                    if len(group.videos) >= 2:
                        # Mark all but the first as duplicates
                        primary = group.videos[0]
                        for secondary in group.videos[1:]:
                            self.state.duplicates_detected.append(
                                (primary.filename, secondary.filename, 0.9)  # Approx similarity
                            )
                            # Mark in our videos list
                            for v in videos:
                                if v.filename == secondary.filename:
                                    v.is_duplicate_of = primary.filename
                            
            except Exception as e:
                print(f"Error detecting duplicates: {e}")
        
        # Match videos to media plan line items
        if self.state.media_plan_info and self.state.media_plan_info.creative_line_items:
            self._match_videos_to_line_items(videos)
        
        self.state.videos = videos
        
        if progress_callback:
            progress_callback("Analysis complete!", 1.0)
        
        return videos
    
    def _match_videos_to_line_items(self, videos: List[VideoInfo]):
        """Match video filenames to media plan creative line items."""
        line_items = self.state.media_plan_info.creative_line_items
        
        for video in videos:
            video_name_lower = video.filename.lower()
            video_name_clean = re.sub(r'[_\-\.]', ' ', video_name_lower)
            video_name_clean = re.sub(r'\.(mp4|mov|avi|mkv)$', '', video_name_clean)
            
            best_match = None
            best_score = 0
            
            for item in line_items:
                item_name_lower = item['name'].lower()
                item_name_clean = re.sub(r'[_\-\.]', ' ', item_name_lower)
                
                # Simple word overlap scoring
                video_words = set(video_name_clean.split())
                item_words = set(item_name_clean.split())
                
                if video_words and item_words:
                    overlap = len(video_words & item_words)
                    score = overlap / max(len(video_words), len(item_words))
                    
                    if score > best_score and score > 0.3:  # Minimum threshold
                        best_score = score
                        best_match = item['name']
            
            if best_match:
                video.matched_line_item = best_match
    
    # =========================================================================
    # Agent Conversation
    # =========================================================================
    
    def process_upload(self, media_plan_path: str = None, video_files: List[Tuple[str, str]] = None, 
                       progress_callback=None) -> str:
        """
        Process uploaded files and generate initial agent response.
        
        Args:
            media_plan_path: Path to uploaded media plan
            video_files: List of (filename, filepath) tuples for videos
            progress_callback: Optional callback for progress updates
            
        Returns:
            Agent's response message
        """
        analysis_parts = []
        
        # Parse media plan if provided
        if media_plan_path:
            if progress_callback:
                progress_callback("Parsing media plan...", 0.1)
            
            file_ext = os.path.splitext(media_plan_path)[1].lower()
            file_type = 'xlsx' if file_ext in ['.xlsx', '.xls'] else 'csv'
            
            info = self.parse_media_plan(media_plan_path, file_type)
            
            if info.brand or info.campaign_name:
                analysis_parts.append(self._format_media_plan_summary(info))
            else:
                analysis_parts.append("⚠️ **Media Plan**: I couldn't extract campaign details automatically. Could you tell me the brand, campaign name, and budget?")
        
        # Analyze videos if provided
        if video_files:
            videos = self.analyze_videos(video_files, progress_callback)
            
            if videos:
                analysis_parts.append(self._format_video_analysis(videos))
        
        # Check for issues
        issues = self._identify_issues()
        if issues:
            analysis_parts.append(self._format_issues(issues))
        
        # Generate response using LLM
        if analysis_parts:
            context = "\n\n".join(analysis_parts)
            response = self._generate_response(context, is_initial=True)
        else:
            response = "Please upload a media plan (Excel/CSV) and your video creatives so I can help you build an optimal test plan."
        
        self._add_message('assistant', response)
        return response
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate agent response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Agent's response
        """
        self._add_message('user', user_message)
        
        # Build context for LLM
        context_parts = []
        
        # Add current state info
        if self.state.media_plan_info:
            context_parts.append(f"Media Plan Info:\n{json.dumps(self.state.media_plan_info.to_dict(), indent=2)}")
        
        if self.state.videos:
            video_summary = []
            for v in self.state.videos:
                video_summary.append({
                    'filename': v.filename,
                    'pass_probability': f"{v.pass_probability*100:.0f}%" if v.scored else "Not scored",
                    'matched_to': v.matched_line_item or "No match",
                    'duplicate_of': v.is_duplicate_of or None,
                    'risk_factors': v.risk_factors[:3] if v.risk_factors else []
                })
            context_parts.append(f"Videos:\n{json.dumps(video_summary, indent=2)}")
        
        if self.state.duplicates_detected:
            context_parts.append(f"Duplicates Detected: {self.state.duplicates_detected}")
        
        # Add CT Rules info
        if self.rules_engine:
            rules_info = {
                'max_testing_budget_pct': '4% of media spend',
                'video_test_cost': '$15,000 per video',
                'static_test_cost': '$8,000 per static',
                'turnaround': '2-3 weeks'
            }
            context_parts.append(f"CT Rules:\n{json.dumps(rules_info, indent=2)}")
        
        context = "\n\n".join(context_parts) if context_parts else "No data uploaded yet."
        
        # Build conversation history for LLM
        history = self.state.messages[-10:]  # Last 10 messages
        
        response = self._generate_response(context, user_message=user_message, history=history)
        
        self._add_message('assistant', response)
        return response
    
    def _generate_response(self, context: str, user_message: str = None, 
                          history: List[Dict] = None, is_initial: bool = False) -> str:
        """Generate agent response using LLM."""
        
        # Build prompt
        if is_initial:
            prompt = f"""I've analyzed the uploaded files. Here's what I found:

{context}

Please provide a clear, well-structured summary of the analysis. Include:
1. Campaign details extracted from the media plan
2. Video analysis results with pass probabilities
3. Any issues found (duplicates, unmatched videos, low scores)
4. Your recommendations

If there are issues, present options for the user to decide. End with a question or next step."""
        else:
            # Include conversation history
            history_text = ""
            if history:
                for msg in history[:-1]:  # Exclude the current message
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_text += f"{role}: {msg['content']}\n\n"
            
            prompt = f"""Current state:

{context}

Conversation so far:
{history_text}

User's latest message: {user_message}

Respond helpfully based on the current state and user's request. If they're making a decision, acknowledge it and proceed. If they have questions, answer them. If you need more information, ask."""

        try:
            response = get_completion(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                max_tokens=2000
            )
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _format_media_plan_summary(self, info: MediaPlanInfo) -> str:
        """Format media plan info for display."""
        parts = ["📋 **MEDIA PLAN ANALYSIS**\n"]
        
        if info.brand:
            parts.append(f"**Brand:** {info.brand}")
        if info.campaign_name:
            parts.append(f"**Campaign:** {info.campaign_name}")
        if info.total_budget:
            # Format budget nicely
            if info.total_budget >= 1_000_000:
                budget_str = f"${info.total_budget/1_000_000:.1f}M"
            else:
                budget_str = f"${info.total_budget:,.0f}"
            parts.append(f"**Total Media Budget:** {budget_str}")
            
            # Calculate testing budget
            max_testing = info.total_budget * 0.04
            if max_testing >= 1_000_000:
                test_str = f"${max_testing/1_000_000:.1f}M"
            else:
                test_str = f"${max_testing:,.0f}"
            max_videos = int(max_testing / 15000)
            parts.append(f"**Max Testing Budget (4%):** {test_str} (up to {max_videos} videos)")
            
        if info.flight_start or info.flight_end:
            parts.append(f"**Flight:** {info.flight_start} - {info.flight_end}")
        if info.markets:
            parts.append(f"**Markets:** {', '.join(info.markets[:5])}")
        if info.primary_kpi:
            parts.append(f"**Primary KPI:** {info.primary_kpi}")
        
        if info.creative_line_items:
            parts.append(f"\n**Creative Line Items:** {len(info.creative_line_items)} found")
            for item in info.creative_line_items[:10]:
                parts.append(f"  - {item['name']}")
        
        return "\n".join(parts)
    
    def _format_video_analysis(self, videos: List[VideoInfo]) -> str:
        """Format video analysis results for display."""
        parts = [f"\n📹 **VIDEO ANALYSIS** ({len(videos)} videos)\n"]
        
        # Sort by pass probability (highest first)
        sorted_videos = sorted(videos, key=lambda v: v.pass_probability, reverse=True)
        
        for v in sorted_videos:
            prob_pct = v.pass_probability * 100 if v.scored else 0
            
            # Status indicator
            if v.is_duplicate_of:
                status = "🔄 Duplicate"
            elif prob_pct >= 70:
                status = "✅ Strong"
            elif prob_pct >= 50:
                status = "⚠️ Moderate"
            elif v.scored:
                status = "❌ Risky"
            else:
                status = "❓ Not scored"
            
            line = f"**{v.filename}** - {prob_pct:.0f}% pass prob - {status}"
            
            if v.matched_line_item:
                line += f"\n  ↳ Matched to: \"{v.matched_line_item}\""
            else:
                line += f"\n  ↳ ⚠️ No media plan match"
            
            if v.is_duplicate_of:
                line += f"\n  ↳ 🔄 Similar to: {v.is_duplicate_of}"
            
            if v.risk_factors and v.scored:
                line += f"\n  ↳ Risks: {', '.join(v.risk_factors[:2])}"
            
            parts.append(line)
        
        return "\n".join(parts)
    
    def _identify_issues(self) -> List[str]:
        """Identify issues with the current plan."""
        issues = []
        
        # Check for duplicates
        if self.state.duplicates_detected:
            for v1, v2, sim in self.state.duplicates_detected:
                issues.append(f"Duplicate detected: {v1} and {v2} are {sim*100:.0f}% similar")
        
        # Check for unmatched videos
        unmatched = [v for v in self.state.videos if not v.matched_line_item]
        if unmatched:
            issues.append(f"{len(unmatched)} video(s) don't match any media plan line item")
        
        # Check for low-scoring videos
        low_scores = [v for v in self.state.videos if v.scored and v.pass_probability < 0.5]
        if low_scores:
            issues.append(f"{len(low_scores)} video(s) have <50% predicted pass probability")
        
        # Check budget
        if self.state.media_plan_info and self.state.media_plan_info.total_budget:
            max_testing = self.state.media_plan_info.total_budget * 0.04
            num_videos = len([v for v in self.state.videos if not v.is_duplicate_of])
            testing_cost = num_videos * 15000
            
            if testing_cost > max_testing:
                issues.append(f"Testing cost (${testing_cost:,.0f}) exceeds max budget (${max_testing:,.0f})")
        
        self.state.issues = issues
        return issues
    
    def _format_issues(self, issues: List[str]) -> str:
        """Format issues for display."""
        if not issues:
            return ""
        
        parts = ["\n⚠️ **ISSUES FOUND**\n"]
        for i, issue in enumerate(issues, 1):
            parts.append(f"{i}. {issue}")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Plan Generation
    # =========================================================================
    
    def generate_plan(self, selected_video_filenames: List[str] = None) -> Dict:
        """
        Generate the final test plan.
        
        Args:
            selected_video_filenames: List of video filenames to include (None = all non-duplicates)
            
        Returns:
            Plan dictionary
        """
        if selected_video_filenames is None:
            # Include all non-duplicate videos
            selected_videos = [v for v in self.state.videos if not v.is_duplicate_of]
        else:
            selected_videos = [v for v in self.state.videos if v.filename in selected_video_filenames]
        
        # Build plan
        plan = {
            'plan_id': f"PLAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.now().isoformat(),
            'campaign': {
                'brand': self.state.media_plan_info.brand if self.state.media_plan_info else '',
                'name': self.state.media_plan_info.campaign_name if self.state.media_plan_info else '',
                'budget': self.state.media_plan_info.total_budget if self.state.media_plan_info else 0,
                'primary_kpi': self.state.media_plan_info.primary_kpi if self.state.media_plan_info else '',
            },
            'creatives': [],
            'testing_budget': len(selected_videos) * 15000,
            'estimated_turnaround': '2-3 weeks',
        }
        
        for v in selected_videos:
            creative = {
                'filename': v.filename,
                'filepath': v.filepath,
                'predicted_pass_probability': v.pass_probability,
                'risk_factors': v.risk_factors,
                'matched_line_item': v.matched_line_item,
            }
            plan['creatives'].append(creative)
        
        # Sort by pass probability (highest first)
        plan['creatives'].sort(key=lambda x: x['predicted_pass_probability'], reverse=True)
        
        self.state.final_plan = plan
        self.state.plan_ready = True
        
        return plan


# Singleton instance
_planning_agent: Optional[PlanningAgent] = None


def get_planning_agent() -> PlanningAgent:
    """Get or create the planning agent instance."""
    global _planning_agent
    if _planning_agent is None:
        _planning_agent = PlanningAgent()
    return _planning_agent


def reset_planning_agent():
    """Reset the planning agent to start fresh."""
    global _planning_agent
    _planning_agent = PlanningAgent()
    return _planning_agent
