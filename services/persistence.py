"""
Persistence Service - Handles saving and loading all app data to disk.

Provides persistence for:
- CT Plans (draft and approved)
- Test Results
- Chat History
- Campaign Metadata
- App Settings
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import pandas as pd


class PersistenceService:
    """
    Centralized persistence service for the CT Orchestrator app.
    
    Directory structure:
    data/
    ├── plans/
    │   ├── draft/          # Plans in progress
    │   └── approved/       # Approved plans
    ├── results/
    │   ├── raw/            # Original uploaded CSVs
    │   └── analyzed/       # Analysis results as JSON
    ├── chat_history/       # Conversation logs
    ├── campaigns.json      # Campaign index/metadata
    └── settings.json       # App settings
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize persistence service.
        
        Args:
            base_dir: Base directory for all data. Defaults to 'data/' in app root.
        """
        if base_dir is None:
            # Default to 'data/' directory relative to the app
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        self.base_dir = Path(base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_dir,
            self.base_dir / 'plans' / 'draft',
            self.base_dir / 'plans' / 'approved',
            self.base_dir / 'results' / 'raw',
            self.base_dir / 'results' / 'analyzed',
            self.base_dir / 'chat_history',
            self.base_dir / 'video_features',  # For ML training
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Campaign Management
    # =========================================================================
    
    def _get_campaigns_index_path(self) -> Path:
        return self.base_dir / 'campaigns.json'
    
    def _load_campaigns_index(self) -> dict:
        """Load the campaigns index."""
        path = self._get_campaigns_index_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {'campaigns': {}}
    
    def _save_campaigns_index(self, index: dict):
        """Save the campaigns index."""
        with open(self._get_campaigns_index_path(), 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    def list_campaigns(self) -> list[dict]:
        """List all campaigns with their metadata."""
        index = self._load_campaigns_index()
        campaigns = []
        for campaign_id, metadata in index.get('campaigns', {}).items():
            campaigns.append({
                'id': campaign_id,
                **metadata
            })
        return sorted(campaigns, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    def get_campaign(self, campaign_id: str) -> Optional[dict]:
        """Get campaign metadata by ID."""
        index = self._load_campaigns_index()
        if campaign_id in index.get('campaigns', {}):
            return {'id': campaign_id, **index['campaigns'][campaign_id]}
        return None
    
    def save_campaign_metadata(self, campaign_id: str, metadata: dict):
        """Save or update campaign metadata."""
        index = self._load_campaigns_index()
        if 'campaigns' not in index:
            index['campaigns'] = {}
        
        metadata['updated_at'] = datetime.now().isoformat()
        if campaign_id not in index['campaigns']:
            metadata['created_at'] = datetime.now().isoformat()
        
        index['campaigns'][campaign_id] = metadata
        self._save_campaigns_index(index)
    
    def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign and all associated data."""
        index = self._load_campaigns_index()
        if campaign_id not in index.get('campaigns', {}):
            return False
        
        # Delete associated files
        # Plans
        for plan_type in ['draft', 'approved']:
            plan_dir = self.base_dir / 'plans' / plan_type
            for f in plan_dir.glob(f'{campaign_id}_*.json'):
                f.unlink()
        
        # Results
        for result_type in ['raw', 'analyzed']:
            result_dir = self.base_dir / 'results' / result_type
            for f in result_dir.glob(f'{campaign_id}_*'):
                if f.is_file():
                    f.unlink()
        
        # Chat history
        chat_file = self.base_dir / 'chat_history' / f'{campaign_id}.json'
        if chat_file.exists():
            chat_file.unlink()
        
        # Remove from index
        del index['campaigns'][campaign_id]
        self._save_campaigns_index(index)
        return True
    
    # =========================================================================
    # CT Plans
    # =========================================================================
    
    def save_plan(self, campaign_id: str, plan_data: dict, is_approved: bool = False) -> str:
        """
        Save a CT plan.
        
        Args:
            campaign_id: Campaign identifier
            plan_data: Plan data dictionary
            is_approved: Whether this is an approved plan or draft
            
        Returns:
            Plan ID
        """
        plan_type = 'approved' if is_approved else 'draft'
        plan_dir = self.base_dir / 'plans' / plan_type
        
        # Generate plan ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plan_id = f"{campaign_id}_{timestamp}"
        
        # Add metadata
        plan_data['_meta'] = {
            'plan_id': plan_id,
            'campaign_id': campaign_id,
            'saved_at': datetime.now().isoformat(),
            'is_approved': is_approved,
        }
        
        # Save plan
        plan_path = plan_dir / f'{plan_id}.json'
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2, default=str)
        
        # Update campaign metadata
        campaign_meta = self.get_campaign(campaign_id) or {}
        campaign_meta['name'] = plan_data.get('campaign', {}).get('name', 'Unknown')
        campaign_meta['brand'] = plan_data.get('campaign', {}).get('brand', {}).get('name', 'Unknown')
        if is_approved:
            campaign_meta['has_approved_plan'] = True
            campaign_meta['approved_plan_id'] = plan_id
        else:
            campaign_meta['has_draft_plan'] = True
            campaign_meta['draft_plan_id'] = plan_id
        
        self.save_campaign_metadata(campaign_id, campaign_meta)
        
        return plan_id
    
    def load_plan(self, plan_id: str, is_approved: bool = False) -> Optional[dict]:
        """Load a CT plan by ID."""
        plan_type = 'approved' if is_approved else 'draft'
        plan_path = self.base_dir / 'plans' / plan_type / f'{plan_id}.json'
        
        if plan_path.exists():
            with open(plan_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_plans(self, campaign_id: str = None, is_approved: bool = None) -> list[dict]:
        """
        List plans, optionally filtered by campaign and/or approval status.
        """
        plans = []
        
        plan_types = []
        if is_approved is None:
            plan_types = ['draft', 'approved']
        elif is_approved:
            plan_types = ['approved']
        else:
            plan_types = ['draft']
        
        for plan_type in plan_types:
            plan_dir = self.base_dir / 'plans' / plan_type
            pattern = f'{campaign_id}_*.json' if campaign_id else '*.json'
            
            for plan_path in plan_dir.glob(pattern):
                with open(plan_path, 'r') as f:
                    plan_data = json.load(f)
                    plans.append({
                        'plan_id': plan_path.stem,
                        'campaign_id': plan_data.get('_meta', {}).get('campaign_id'),
                        'campaign_name': plan_data.get('campaign', {}).get('name', 'Unknown'),
                        'is_approved': plan_type == 'approved',
                        'saved_at': plan_data.get('_meta', {}).get('saved_at'),
                        'creative_count': len(plan_data.get('all_creatives', [])),
                    })
        
        return sorted(plans, key=lambda x: x.get('saved_at', ''), reverse=True)
    
    def delete_plan(self, plan_id: str, is_approved: bool = False) -> bool:
        """Delete a plan."""
        plan_type = 'approved' if is_approved else 'draft'
        plan_path = self.base_dir / 'plans' / plan_type / f'{plan_id}.json'
        
        if plan_path.exists():
            plan_path.unlink()
            return True
        return False
    
    # =========================================================================
    # Test Results
    # =========================================================================
    
    def save_results(self, campaign_id: str, results_data: dict, raw_csv_content: bytes = None) -> str:
        """
        Save test results.
        
        Args:
            campaign_id: Campaign identifier
            results_data: Parsed and analyzed results
            raw_csv_content: Original CSV file content (optional)
            
        Returns:
            Results ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_id = f"{campaign_id}_{timestamp}"
        
        # Save raw CSV if provided
        if raw_csv_content:
            raw_path = self.base_dir / 'results' / 'raw' / f'{results_id}.csv'
            with open(raw_path, 'wb') as f:
                f.write(raw_csv_content)
        
        # Add metadata
        results_data['_meta'] = {
            'results_id': results_id,
            'campaign_id': campaign_id,
            'saved_at': datetime.now().isoformat(),
            'has_raw_csv': raw_csv_content is not None,
        }
        
        # Save analyzed results
        analyzed_path = self.base_dir / 'results' / 'analyzed' / f'{results_id}.json'
        with open(analyzed_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Update campaign metadata
        campaign_meta = self.get_campaign(campaign_id) or {}
        campaign_meta['has_results'] = True
        campaign_meta['latest_results_id'] = results_id
        if 'results_ids' not in campaign_meta:
            campaign_meta['results_ids'] = []
        campaign_meta['results_ids'].append(results_id)
        self.save_campaign_metadata(campaign_id, campaign_meta)
        
        return results_id
    
    def load_results(self, results_id: str) -> Optional[dict]:
        """Load test results by ID."""
        analyzed_path = self.base_dir / 'results' / 'analyzed' / f'{results_id}.json'
        
        if analyzed_path.exists():
            with open(analyzed_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_raw_csv(self, results_id: str) -> Optional[pd.DataFrame]:
        """Load the raw CSV file for results."""
        raw_path = self.base_dir / 'results' / 'raw' / f'{results_id}.csv'
        
        if raw_path.exists():
            return pd.read_csv(raw_path)
        return None
    
    def list_results(self, campaign_id: str = None) -> list[dict]:
        """List results, optionally filtered by campaign."""
        results = []
        analyzed_dir = self.base_dir / 'results' / 'analyzed'
        pattern = f'{campaign_id}_*.json' if campaign_id else '*.json'
        
        for results_path in analyzed_dir.glob(pattern):
            with open(results_path, 'r') as f:
                results_data = json.load(f)
                meta = results_data.get('_meta', {})
                results.append({
                    'results_id': results_path.stem,
                    'campaign_id': meta.get('campaign_id'),
                    'saved_at': meta.get('saved_at'),
                    'has_raw_csv': meta.get('has_raw_csv', False),
                })
        
        return sorted(results, key=lambda x: x.get('saved_at', ''), reverse=True)
    
    # =========================================================================
    # Chat History
    # =========================================================================
    
    def save_chat_history(self, campaign_id: str, messages: list[dict]):
        """
        Save chat history for a campaign.
        
        Args:
            campaign_id: Campaign identifier (use 'global' for non-campaign chats)
            messages: List of message dictionaries with 'role' and 'content'
        """
        chat_path = self.base_dir / 'chat_history' / f'{campaign_id}.json'
        
        chat_data = {
            'campaign_id': campaign_id,
            'updated_at': datetime.now().isoformat(),
            'messages': messages,
        }
        
        with open(chat_path, 'w') as f:
            json.dump(chat_data, f, indent=2, default=str)
    
    def load_chat_history(self, campaign_id: str) -> list[dict]:
        """Load chat history for a campaign."""
        chat_path = self.base_dir / 'chat_history' / f'{campaign_id}.json'
        
        if chat_path.exists():
            with open(chat_path, 'r') as f:
                data = json.load(f)
                return data.get('messages', [])
        return []
    
    def append_chat_message(self, campaign_id: str, role: str, content: str):
        """Append a single message to chat history."""
        messages = self.load_chat_history(campaign_id)
        messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
        })
        self.save_chat_history(campaign_id, messages)
    
    def list_chat_histories(self) -> list[dict]:
        """List all chat histories."""
        histories = []
        chat_dir = self.base_dir / 'chat_history'
        
        for chat_path in chat_dir.glob('*.json'):
            with open(chat_path, 'r') as f:
                data = json.load(f)
                histories.append({
                    'campaign_id': data.get('campaign_id'),
                    'updated_at': data.get('updated_at'),
                    'message_count': len(data.get('messages', [])),
                })
        
        return sorted(histories, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    # =========================================================================
    # Settings
    # =========================================================================
    
    def _get_settings_path(self) -> Path:
        return self.base_dir / 'settings.json'
    
    def load_settings(self) -> dict:
        """Load app settings."""
        path = self._get_settings_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_settings(self, settings: dict):
        """Save app settings."""
        with open(self._get_settings_path(), 'w') as f:
            json.dump(settings, f, indent=2)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a single setting value."""
        settings = self.load_settings()
        return settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a single setting value."""
        settings = self.load_settings()
        settings[key] = value
        self.save_settings(settings)
    
    # =========================================================================
    # Export / Import
    # =========================================================================
    
    def export_campaign(self, campaign_id: str, export_path: str) -> bool:
        """
        Export all data for a campaign to a zip file.
        """
        import zipfile
        
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return False
        
        with zipfile.ZipFile(export_path, 'w') as zf:
            # Campaign metadata
            zf.writestr('campaign.json', json.dumps(campaign, indent=2, default=str))
            
            # Plans
            for plan_type in ['draft', 'approved']:
                plan_dir = self.base_dir / 'plans' / plan_type
                for plan_path in plan_dir.glob(f'{campaign_id}_*.json'):
                    zf.write(plan_path, f'plans/{plan_type}/{plan_path.name}')
            
            # Results
            for result_type in ['raw', 'analyzed']:
                result_dir = self.base_dir / 'results' / result_type
                for result_path in result_dir.glob(f'{campaign_id}_*'):
                    zf.write(result_path, f'results/{result_type}/{result_path.name}')
            
            # Chat history
            chat_path = self.base_dir / 'chat_history' / f'{campaign_id}.json'
            if chat_path.exists():
                zf.write(chat_path, f'chat_history/{chat_path.name}')
        
        return True
    
    def export_all(self, export_path: str) -> bool:
        """Export all app data to a zip file."""
        shutil.make_archive(export_path.replace('.zip', ''), 'zip', self.base_dir)
        return True
    
    # =========================================================================
    # Video Features (for ML Training)
    # =========================================================================
    
    def save_video_features(self, creative_id: str, features: dict) -> bool:
        """
        Save extracted video features for a creative.
        
        Args:
            creative_id: Creative identifier
            features: Dict of extracted features
            
        Returns:
            True if saved successfully
        """
        path = self.base_dir / 'video_features' / f'{creative_id}.json'
        try:
            with open(path, 'w') as f:
                json.dump(features, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving video features: {e}")
            return False
    
    def load_video_features(self, creative_id: str) -> Optional[dict]:
        """
        Load video features for a creative.
        
        Args:
            creative_id: Creative identifier
            
        Returns:
            Features dict or None if not found
        """
        path = self.base_dir / 'video_features' / f'{creative_id}.json'
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_video_features(self) -> list:
        """
        List all creatives with video features.
        
        Returns:
            List of creative_ids that have features stored
        """
        features_dir = self.base_dir / 'video_features'
        if not features_dir.exists():
            return []
        return [f.stem for f in features_dir.glob('*.json')]
    
    def load_all_video_features(self) -> list:
        """
        Load all video features for ML training.
        
        Returns:
            List of feature dicts
        """
        all_features = []
        for creative_id in self.list_video_features():
            features = self.load_video_features(creative_id)
            if features:
                all_features.append(features)
        return all_features


# Singleton instance
_persistence_service = None


def get_persistence_service(base_dir: str = None) -> PersistenceService:
    """Get the singleton persistence service instance."""
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = PersistenceService(base_dir)
    return _persistence_service
