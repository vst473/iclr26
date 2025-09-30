#!/usr/bin/env python3
"""
DataHub JSON Data Insertion Script

This script provides functionality to insert book/document metadata into DataHub
with proper formatting, tagging, ownership, and browse paths support.

Usage:
    - Instantiate DataHubBookInserter with your DataHub host
    - Call insert_record() or insert_batch() with JSON book records
    - Records are transformed into DataHub MCPs with rich metadata
"""
import time 
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re
from datahub.emitter.mcp import MetadataChangeProposalWrapper
import datahub.emitter.mce_builder as builder
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.utilities.urns.dataset_urn import DatasetUrn
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    GlobalTagsClass,
    TagAssociationClass,
    SubTypesClass,
    DatasetSnapshotClass,
    MetadataChangeEventClass,
    OwnershipClass,
    OwnerClass,
    OwnershipTypeClass,
    StatusClass,
    DataPlatformInstanceClass,
    BrowsePathsClass,
    EditableDatasetPropertiesClass,
    InstitutionalMemoryClass,
    InstitutionalMemoryMetadataClass,
    AuditStampClass
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataHubBookInserter:
    """
    Class to handle insertion of book metadata into DataHub with proper
    formatting, tagging, ownership, and browse paths.
    """
    
    def __init__(self, datahub_host: str = "http://localhost:8080", 
                 platform: str = "s3", database: str = "books", schema: str = "archive"):
        """
        Initialize DataHub emitter and set platform/namespace configuration
        
        Args:
            datahub_host: DataHub GMS host URL
            platform: Platform identifier (default: s3)
            database: Database/namespace name (default: books)
            schema: Schema/collection name (default: archive)
        """
        self.emitter = DatahubRestEmitter(gms_server=datahub_host)
        self.platform = platform
        self.database = database
        self.schema = schema
        
        # Validate platform on initialization
        if not self.validate_platform():
            logger.warning(f"Platform '{self.platform}' may not be recognized in DataHub")
        
    def validate_platform(self) -> bool:
        """
        Validate that the platform exists in DataHub
        
        Returns:
            bool: True if platform is valid
        """
        # Common built-in DataHub platforms
        built_in_platforms = ["s3", "bigquery", "snowflake", "kafka", "mysql", 
                             "postgres", "hive", "redshift", "mongodb"]
        
        # Simple validation - could be enhanced with actual platform lookup
        return bool(self.platform and (self.platform in built_in_platforms or 
                                      not self.platform.isspace()))
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for safe usage in URNs and identifiers
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text string
        """
        if not text:
            return ""
            
        # Remove special characters and normalize whitespace
        cleaned = re.sub(r'[^a-zA-Z0-9_\s]', '', text)
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned.lower()
    
    def clean_and_validate_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate record data to ensure it has required fields and proper types
        
        Args:
            record: Raw record data from JSON
            
        Returns:
            Dict[str, Any]: Cleaned and validated record data
        
        Raises:
            ValueError: If record is missing critical fields after cleaning
        """
        # Handle nested fields structure
        fields = record.get('fields', record)
        
        # Extract file info if present
        files = fields.get('files', [])
        
        # Clean and validate required fields
        cleaned = {
            'identifier': fields.get('identifier', '').strip(),
            'title': fields.get('title', '').strip(),
            'description': fields.get('description', '').strip(),
            'subject': fields.get('subject', []),
            'creator': fields.get('creator', []),
            'collection': fields.get('collection', []),
            'date': fields.get('date', ''),
            'year': int(fields.get('year', 0)),
            'language': fields.get('language', []),
            'mediatype': fields.get('mediatype', '').strip(),
            'num_favorites': int(fields.get('num_favorites', 0)),
            'downloads': int(fields.get('downloads', 0)),
            'files_count': int(fields.get('files_count', 0)),
            'item_size': int(fields.get('item_size', 0)),
        }
        
        # Add file information if available
        if files and len(files) > 0:
            first_file = files[0]
            cleaned.update({
                'file_name': first_file.get('file_name', ''),
                'presigned_url': first_file.get('presigned_url', ''),
                's3_key': first_file.get('s3_key', '')
            })
        
        # Ensure lists are actually lists
        for list_field in ['subject', 'creator', 'collection', 'language']:
            if not isinstance(cleaned[list_field], list):
                cleaned[list_field] = [cleaned[list_field]] if cleaned[list_field] else []
        
        # Validate critical fields
        if not cleaned['identifier']:
            raise ValueError("Record is missing required identifier field")
        
        if not cleaned['title']:
            cleaned['title'] = f"Untitled ({cleaned['identifier']})"
            logger.warning(f"Missing title for {cleaned['identifier']}, using default")
                
        return cleaned
    
    def create_dataset_urn(self, identifier: str) -> str:
        """
        Create DataHub URN for the dataset using proper platform and hierarchy
        
        Args:
            identifier: Unique identifier for the book
            
        Returns:
            str: DataHub URN string for the dataset
            
        Raises:
            ValueError: If identifier is empty after cleaning
        """
        # Sanitize identifier for URN
        clean_identifier = self.clean_text(identifier)
        if not clean_identifier:
            raise ValueError("Empty identifier after cleaning")
        
        # Format depends on platform
        if self.platform.lower() == "s3":
            # For S3, use format: bucket/key_path
            # The database becomes the bucket name, schema becomes a prefix
            name = f"{self.database}/{self.schema}/{clean_identifier}"
        else:
            # For other platforms, use standard database.schema.table format
            name = f"{self.database}.{self.schema}.{clean_identifier}"
        
        # Use DataHub's builder pattern for proper URN construction
        return builder.make_dataset_urn(
            platform=self.platform,
            name=name,
            env="PROD"
        )
    
    def create_browse_paths(self, collections: List[str]) -> BrowsePathsClass:
        """
        Create browse paths for hierarchical navigation in DataHub UI
        
        Args:
            collections: List of collections the book belongs to
            
        Returns:
            BrowsePathsClass: Browse paths aspect for DataHub
        """
        paths = []
        
        # Base path: /platform/database/schema
        base_path = f"/{self.platform}/{self.database}/{self.schema}"
        paths.append(base_path)
        
        # Add collection-specific paths
        for collection in collections:
            if collection and collection.strip():
                clean_collection = self.clean_text(collection)
                if clean_collection:
                    collection_path = f"{base_path}/{clean_collection}"
                    paths.append(collection_path)
        
        return BrowsePathsClass(paths=paths)

    def create_editable_properties(self, record: Dict[str, Any]) -> Optional[EditableDatasetPropertiesClass]:
        """
        Create editable dataset properties with documentation and URLs
        
        Args:
            record: Cleaned record data
            
        Returns:
            Optional[EditableDatasetPropertiesClass]: Editable properties with URLs
        """
        # Create documentation with file access information
        documentation = ""
        
        # Add basic description
        if record.get('description'):
            documentation += f"{record['description']}\n\n"
        
        # Add file access information
        if record.get('presigned_url'):
            documentation += f"**Direct Access URL:** [View/Download File]({record['presigned_url']})\n\n"
        
        if record.get('file_name'):
            documentation += f"**File Name:** {record['file_name']}\n"
        
        if record.get('s3_key'):
            documentation += f"**S3 Location:** {record['s3_key']}\n"
        
        # Add metadata
        if record.get('downloads', 0) > 0:
            documentation += f"**Downloads:** {record['downloads']:,}\n"
        
        if record.get('num_favorites', 0) > 0:
            documentation += f"**Favorites:** {record['num_favorites']:,}\n"
        
        if record.get('item_size', 0) > 0:
            size_mb = record['item_size'] / (1024 * 1024)
            documentation += f"**File Size:** {size_mb:.2f} MB\n"
        
        return EditableDatasetPropertiesClass(
            description=documentation.strip()
        ) if documentation.strip() else None
    
    def create_institutional_memory(self, record: Dict[str, Any]) -> Optional[InstitutionalMemoryClass]:
        """
        Create institutional memory with direct links to files
        
        Args:
            record: Cleaned record data
            
        Returns:
            Optional[InstitutionalMemoryClass]: Institutional memory with file links
        """
        elements = []
        
        # Create audit stamp for metadata
        current_time = int(datetime.now().timestamp() * 1000)  # milliseconds since epoch
        audit_stamp = AuditStampClass(
            time=current_time,
            actor=builder.make_user_urn("system")
        )
        
        # Add direct file access link
        if record.get('presigned_url'):
            # Clean the presigned URL to make it more permanent (remove query params if needed)
            clean_url = record['presigned_url'].split('?')[0] if '?' in record['presigned_url'] else record['presigned_url']
            
            elements.append(InstitutionalMemoryMetadataClass(
                url=record['presigned_url'],  # Use the full presigned URL for actual access
                description=f"Direct access to {record.get('file_name', 'file')}",
                createStamp=audit_stamp
            ))
        
        # Add archive.org link if identifier is available
        if record.get('identifier'):
            archive_url = f"https://archive.org/details/{record['identifier']}"
            elements.append(InstitutionalMemoryMetadataClass(
                url=archive_url,
                description="View on Archive.org",
                createStamp=audit_stamp
            ))
        
        return InstitutionalMemoryClass(elements=elements) if elements else None

    def create_tags(self, subjects: List[str], collections: List[str], 
                   language: List[str], mediatype: str) -> Optional[GlobalTagsClass]:
        """
        Create tags from subjects, collections, language, and mediatype
        
        Args:
            subjects: List of subject categories
            collections: List of collections
            language: List of languages
            mediatype: Media type
            
        Returns:
            Optional[GlobalTagsClass]: Tags aspect for DataHub, or None if no tags created
        """
        tags = []
        
        # Add subject tags with prefix
        for subject in subjects:
            if subject and subject.strip():
                tag_name = f"subject_{self.clean_text(subject)[:50]}"
                tags.append(TagAssociationClass(tag=f"urn:li:tag:{tag_name}"))
        
        # Add collection tags
        for collection in collections:
            if collection and collection.strip():
                tag_name = f"collection_{self.clean_text(collection)[:50]}"
                tags.append(TagAssociationClass(tag=f"urn:li:tag:{tag_name}"))
        
        # Add language tags
        for lang in language:
            if lang and lang.strip():
                tag_name = f"language_{self.clean_text(lang)[:10]}"
                tags.append(TagAssociationClass(tag=f"urn:li:tag:{tag_name}"))
        
        # Add mediatype tag
        if mediatype and mediatype.strip():
            tag_name = f"mediatype_{self.clean_text(mediatype)[:20]}"
            tags.append(TagAssociationClass(tag=f"urn:li:tag:{tag_name}"))
        
        return GlobalTagsClass(tags=tags) if tags else None
    
    def create_ownership(self, creators: List[str]) -> Optional[OwnershipClass]:
        """
        Create ownership information from creators/authors
        
        Args:
            creators: List of creators/authors
            
        Returns:
            Optional[OwnershipClass]: Ownership aspect for DataHub, or None if no owners
        """
        if not creators:
            return None
            
        owners = []
        for creator in creators:
            if creator and creator.strip():
                clean_creator = self.clean_text(creator)[:50]
                if clean_creator:
                    # Create a user URN for the creator
                    owner_urn = f"urn:li:corpuser:{clean_creator}"
                    owners.append(OwnerClass(
                        owner=owner_urn,
                        type=OwnershipTypeClass.PRODUCER  # Authors are producers
                    ))
        
        return OwnershipClass(owners=owners) if owners else None
    
    def create_custom_properties(self, record: Dict[str, Any]) -> Dict[str, str]:
        """
        Create custom properties for additional metadata
        
        Args:
            record: Cleaned record data
            
        Returns:
            Dict[str, str]: Dictionary of custom properties for DataHub
        """
        properties = {
            "source": "s3-books-archive",
            "governance": "ENABLED"
        }
        
        # Add statistical information
        if record.get('num_favorites', 0) > 0:
            properties['favorites'] = str(record['num_favorites'])
        if record.get('downloads', 0) > 0:
            properties['downloads'] = str(record['downloads'])
        if record.get('files_count', 0) > 0:
            properties['files_count'] = str(record['files_count'])
        if record.get('item_size', 0) > 0:
            properties['item_size_bytes'] = str(record['item_size'])
        
        # Add year if available
        if record.get('year', 0) > 0:
            properties['year'] = str(record['year'])
            
        # Add date if available
        if record.get('date'):
            properties['date'] = record['date']
        
        # Add fileinfo as a JSON string if present
        if record.get('fileinfo'):
            properties['fileinfo'] = json.dumps(record['fileinfo'])
        
        # Add direct access URL
        if record.get('presigned_url'):
            properties['direct_access_url'] = record['presigned_url']
            
        # Add file name
        if record.get('file_name'):
            properties['file_name'] = record['file_name']
            
        # Add S3 key
        if record.get('s3_key'):
            properties['s3_key'] = record['s3_key']
            
        # Add archive.org URL
        if record.get('identifier'):
            properties['archive_org_url'] = f"https://archive.org/details/{record['identifier']}"
        
        if record.get('creator'):
            properties['creator'] = ', '.join(record['creator'])
        return properties
    
    def create_dataset_properties(self, record: Dict[str, Any]) -> DatasetPropertiesClass:
        """
        Create dataset properties aspect with name, description and custom properties
        
        Args:
            record: Cleaned record data
            
        Returns:
            DatasetPropertiesClass: Dataset properties aspect for DataHub
        """
        custom_properties = self.create_custom_properties(record)
        
        # Ensure title and description aren't too long for DataHub
        title = record['title'][:200] if record['title'] else "Untitled"
        
        # Enhanced description with URL information
        base_description = record['description'] if record['description'] else f"Book: {title} from S3 books archive"
        
        # Add URL information to description if available
        url_info = ""
        if record.get('presigned_url'):
            url_info += f"\n\n🔗 Direct Access: Available via S3"
        if record.get('identifier'):
            url_info += f"\n📚 Archive.org: https://archive.org/details/{record['identifier']}"
        
        description = (base_description + url_info)[:2000]  # Ensure it doesn't exceed DataHub limits
        
        return DatasetPropertiesClass(
            name=title,
            description=description,
            customProperties=custom_properties
        )
    
    def create_all_aspects(self, record: Dict[str, Any]) -> Tuple[str, List[Tuple[str, Any]]]:
        """
        Create all dataset aspects for a record
        
        Args:
            record: Cleaned record data
            
        Returns:
            Tuple[str, List[Tuple[str, Any]]]: Dataset URN and list of aspect tuples (name, object)
        """
        # Create URN
        dataset_urn = self.create_dataset_urn(record['identifier'])
        
        # Create all aspects
        aspects = []
        
        # Always include dataset properties
        dataset_properties = self.create_dataset_properties(record)
        aspects.append(("datasetProperties", dataset_properties))
        
        # Conditionally include tags
        global_tags = self.create_tags(
            record['subject'],
            record['collection'],
            record['language'],
            record['mediatype']
        )
        if global_tags:
            aspects.append(("globalTags", global_tags))
        
        # Conditionally include ownership
        # ownership = self.create_ownership(record['creator'])
        # if ownership:
        #     aspects.append(("ownership", ownership))
        
        # Always include browse paths
        browse_paths = self.create_browse_paths(record['collection'])
        aspects.append(("browsePaths", browse_paths))
        
        # Always include status (active)
        status = StatusClass(removed=False)
        aspects.append(("status", status))
        
        # Conditionally include editable properties with URLs
        editable_properties = self.create_editable_properties(record)
        if editable_properties:
            aspects.append(("editableDatasetProperties", editable_properties))
        
        # Conditionally include institutional memory with direct links
        institutional_memory = self.create_institutional_memory(record)
        if institutional_memory:
            aspects.append(("institutionalMemory", institutional_memory))
        
        return dataset_urn, aspects
    
    def insert_record(self, record: Dict[str, Any]) -> bool:
        """
        Insert a single record into DataHub with full metadata
        
        Args:
            record: Record data to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clean and validate data
            cleaned_record = self.clean_and_validate_data(record)
            
            # Create all aspects
            dataset_urn, aspects = self.create_all_aspects(cleaned_record)
            
            # Create and emit metadata change proposals
            mcps = []
            for aspect_name, aspect in aspects:
                mcps.append(MetadataChangeProposalWrapper(
                    entityType="dataset",
                    entityUrn=dataset_urn,
                    aspectName=aspect_name,
                    aspect=aspect
                ))
            
            # Emit all metadata changes
            for mcp in mcps:
                self.emitter.emit_mcp(mcp)
            
            logger.info(f"Successfully inserted: {cleaned_record['identifier']} - {cleaned_record['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting {record.get('identifier', 'unknown')}: {str(e)}")
            return False
    
    def insert_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Insert multiple records into DataHub with progress reporting
        
        Args:
            records: List of record data
            
        Returns:
            Dict[str, int]: Dictionary with success/failure counts
        """
        results = {'success': 0, 'failed': 0, 'total': len(records)}
        
        for i, record in enumerate(records):

            time.sleep(0.1)  # Simulate processing delay
            logger.info(f"Processing record {i+1}/{len(records)}")
            
            if self.insert_record(record):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file, supporting both JSON Lines and standard JSON formats
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List[Dict[str, Any]]: List of parsed records
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip()
        
    # First try to parse as a JSON array
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # Try parsing as JSON Lines format (one JSON object per line)
        records = []
        for line_num, line in enumerate(data.split('\n'), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num}: {line[:100]}...")
        
        if not records:
            raise json.JSONDecodeError("No valid JSON objects found in file", data, 0)
        
        return records

def main():
    """Main function to run the DataHub insertion process"""
    
    # Configuration
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting DataHub JSON Ingestion Script")
    DATAHUB_HOST = "http://localhost:8080"  # Adjust to your DataHub instance
    
    # read json data from file or use sample data
    with open('/raid/nauman/datahub/final_results.json', 'r') as file:
        
        data =  json.load(file)  # Load each line as a separate JSON object
    
    # Filter out error records and only take successful ones
    sample_data = [
        data[key] for key in data.keys() 
        if data[key].get('status') == 'success'
    ]
    
    try:
        # Initialize inserter
        inserter = DataHubBookInserter(datahub_host=DATAHUB_HOST)
        
        logger.info("Starting DataHub insertion process...")
        results = inserter.insert_batch(sample_data)
        
        # Print results
        logger.info(f"Insertion completed: {results['success']}/{results['total']} successful, "
                   f"{results['failed']}/{results['total']} failed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()