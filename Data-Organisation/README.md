# Data Organisation Module

This module provides comprehensive data organisation capabilities for managing large-scale document collections, specifically designed for LLM training data preparation. It handles both metadata ingestion into DataHub and file uploads to distributed storage systems.

## Overview

The Data Organisation module consists of two main components that work together in a deployed architecture:

1. **DataHub Metadata Ingestion** (`ingestion.py`) - Structured metadata management and lineage tracking
2. **JuiceFS Storage Management** (`ingestion_lake.py`) - Distributed file storage and URL generation

### Deployed Architecture: JuiceFS + DataHub Integration

We have deployed a combined **JuiceFS + DataHub** infrastructure that provides comprehensive data discoverability and management:

- **JuiceFS**: Distributed file system for scalable storage of documents, PDFs, and training data
- **DataHub**: Metadata catalog and data discovery platform
- **Combined Benefits**: Rich metadata search capabilities with direct access to stored files

This architecture enables:
- **Metadata-driven Discovery**: Search and filter documents through DataHub's web interface
- **Direct File Access**: Click-through from metadata to actual files via presigned URLs
- **Lineage Tracking**: Full data provenance and transformation history
- **Browse Navigation**: Hierarchical exploration of data collections
- **Rich Search**: Full-text search across metadata fields, tags, and descriptions

## Components

### 1. DataHub Metadata Ingestion (`ingestion.py`)

A robust system for ingesting book/document metadata into DataHub with proper formatting, tagging, ownership, and browse paths support.

#### Key Features

- **Metadata Transformation**: Converts JSON book records into DataHub MCPs (Metadata Change Proposals)
- **Data Validation**: Comprehensive cleaning and validation of record data
- **Rich Metadata Support**: 
  - Dataset properties (title, description, subjects)
  - Global tags for categorization
  - Ownership information
  - Browse paths for hierarchical organization
  - Institutional memory and documentation
- **Batch Processing**: Efficient handling of large record collections
- **Platform Flexibility**: Support for multiple data platforms (S3, BigQuery, Snowflake, etc.)

#### Usage Example

```python
from ingestion import DataHubBookInserter

# Initialize the inserter
inserter = DataHubBookInserter(
    datahub_host="http://localhost:8080",
    platform="s3",
    database="books",
    schema="archive"
)

# Insert a single record
record = {
    "identifier": "book_12345",
    "title": "Example Book Title",
    "description": "Book description",
    "subject": ["Computer Science", "AI"],
    "creator": ["Author Name"],
    "language": ["English"],
    "year": 2023
}

success = inserter.insert_record(record)

# Insert multiple records
records = [record1, record2, record3]
results = inserter.insert_batch(records)
print(f"Processed: {results['success']} successful, {results['failed']} failed")
```

#### Data Structure Requirements

Records should contain the following fields:
- `identifier` (required): Unique identifier for the document
- `title`: Document title
- `description`: Document description
- `subject`: List of subject categories
- `creator`: List of authors/creators
- `collection`: List of collections the document belongs to
- `language`: List of languages
- `year`: Publication year
- `mediatype`: Type of media (book, text, etc.)
- File information (optional):
  - `files`: Array of file objects with `file_name`, `presigned_url`, `s3_key`

### 2. JuiceFS Storage Management (`ingestion_lake.py`)

A high-performance, scalable file upload system designed for distributed storage using JuiceFS with S3-compatible backends.

#### Key Features

- **Parallel Processing**: Multi-process and multi-threaded architecture for optimal performance
- **Batch Operations**: Efficient handling of large file collections
- **Presigned URLs**: Automatic generation of long-term access URLs (5-year default)
- **Error Handling**: Comprehensive error handling and logging
- **Memory Management**: Chunked processing to handle large datasets
- **Configurable**: Flexible configuration for different storage backends

#### Architecture

```
OptimizedJuiceFSUploader
├── ProcessPoolExecutor (CPU parallelization)
│   └── Process per identifier batch
│       └── ThreadPoolExecutor (I/O parallelization)
│           └── Thread per file upload
```

#### Usage Example

```python
from ingestion_lake import OptimizedJuiceFSUploader, UploadConfig

# Configure the uploader
config = UploadConfig(
    endpoint='http://localhost:9000',
    bucket_name='myjfs',
    access_key='your_access_key',
    secret_key='your_secret_key',
    max_workers=16,
    chunk_size=100
)

# Create uploader instance
uploader = OptimizedJuiceFSUploader(config)

# Metadata for organizing uploads
metadata = {
    "language": "hindi",
    "source": "archive",
    "data_type": "books"
}

# Batch upload files
base_dir = "/path/to/your/files"
identifiers = [
    {"identifier": "book1", "title": "Book 1"},
    {"identifier": "book2", "title": "Book 2"}
]

results = uploader.batch_upload_files(base_dir, identifiers, metadata)
```

#### File Organization Structure

Files are organized in the following hierarchy:
```
bucket/
└── data_type/
    └── source/
        └── language/
            └── identifier/
                ├── file1.pdf
                ├── file2.zip
                └── ...
```

Example: `myjfs/books/archive/hindi/book_12345/document.pdf`

## Performance Optimizations

### JuiceFS Uploader Optimizations

1. **Multi-level Parallelism**:
   - Process-level parallelism for identifier batches
   - Thread-level parallelism for file uploads within each identifier

2. **Memory Management**:
   - Chunked processing to avoid memory overflow
   - Configurable chunk sizes based on system resources

3. **I/O Optimization**:
   - Separate S3 client per thread to avoid connection conflicts
   - Concurrent file uploads with ThreadPoolExecutor

4. **Error Recovery**:
   - Individual error handling per file/identifier
   - Comprehensive logging for debugging

### DataHub Ingestion Optimizations

1. **Data Validation**:
   - Comprehensive data cleaning and validation
   - Automatic field type conversion and normalization

2. **Metadata Enrichment**:
   - Automatic generation of browse paths
   - Smart tagging based on content categories
   - Ownership and lineage tracking

3. **Batch Processing**:
   - Efficient batch insertion with progress tracking
   - Rate limiting to prevent DataHub overload

## Configuration

### Environment Variables

For JuiceFS uploads, you can use environment variables:
```bash
export JUICEFS_ENDPOINT=http://localhost:9000
export JUICEFS_BUCKET=myjfs
export JUICEFS_ACCESS_KEY=your_access_key
export JUICEFS_SECRET_KEY=your_secret_key
```

### DataHub Configuration

For DataHub integration:
```bash
export DATAHUB_HOST=http://localhost:8080
export DATAHUB_PLATFORM=s3
export DATAHUB_DATABASE=books
export DATAHUB_SCHEMA=archive
```

## Logging

Both modules use Python's logging framework with configurable levels:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Error Handling

The modules provide comprehensive error handling:

- **File Not Found**: Graceful handling of missing files
- **Network Errors**: Retry mechanisms and timeout handling
- **Data Validation**: Clear error messages for invalid data
- **Resource Limits**: Memory and connection pool management

## Data Discoverability Through JuiceFS + DataHub Deployment

Our deployed infrastructure provides powerful data discoverability capabilities by combining the strengths of both systems:

### How the Integration Works

1. **Data Ingestion Pipeline**:
   ```
   Raw Documents → JuiceFS Storage → Metadata Extraction → DataHub Catalog
                       ↓                                        ↓
                 Presigned URLs ←────────── Search & Discovery
   ```

2. **Discoverability Features**:
   - **Searchable Catalog**: DataHub provides a web interface to search across all document metadata
   - **Faceted Browsing**: Filter by language, subject, collection, creator, year, etc.
   - **Lineage Visualization**: See data transformation and processing history
   - **Direct Access**: Click from search results directly to files stored in JuiceFS
   - **Rich Metadata**: View comprehensive document information including tags, descriptions, and properties

3. **User Workflow**:
   ```
   User searches in DataHub → Finds relevant documents → Clicks presigned URL → Downloads from JuiceFS
   ```

### What This Module Provides

This Data-Organisation module specifically handles the **ingestion side** of the architecture:

- **Metadata Population**: Takes document metadata and populates DataHub catalog
- **File Upload**: Uploads documents to JuiceFS distributed storage
- **URL Generation**: Creates long-term presigned URLs for direct access
- **Hierarchical Organization**: Structures data with proper browse paths and categorization

The module essentially **feeds the discovery system** by ensuring all documents are properly:
- Stored in JuiceFS with organized paths
- Cataloged in DataHub with rich metadata
- Linked together via presigned URLs for seamless access

### Discovery Benefits for LLM Teams

1. **Dataset Curation**: Easily find and select documents for training datasets
2. **Quality Assessment**: Review metadata to understand document characteristics
3. **Access Management**: Secure, time-limited access to training files
4. **Reproducibility**: Track which documents were used in which training runs
5. **Collaboration**: Share discoverable datasets across research teams

## Use Cases

### LLM Training Data Preparation

1. **Large-scale Document Processing**:
   - Process thousands of documents in parallel
   - Maintain metadata lineage and provenance
   - Generate accessible URLs for training pipelines

2. **Multi-language Support**:
   - Organize content by language
   - Maintain language-specific metadata
   - Support for Indic languages and scripts

3. **Quality Assurance**:
   - Data validation and cleaning
   - Error tracking and reporting
   - Audit trails through DataHub

### Data Lake Management

1. **Hierarchical Organization**:
   - Structured file organization
   - Metadata-driven discovery
   - Browse path navigation

2. **Access Management**:
   - Long-term presigned URLs
   - Secure access patterns
   - Audit logging

## Dependencies

### Core Dependencies
- `datahub-kafka`: DataHub Python SDK
- `boto3`: AWS/S3 client library
- `concurrent.futures`: Parallel processing
- `multiprocessing`: Process pool management

### Installation
```bash
pip install datahub-kafka boto3
```

## Best Practices

1. **Resource Management**:
   - Tune `max_workers` based on system capabilities
   - Use appropriate `chunk_size` for memory management
   - Monitor system resources during large uploads

2. **Data Quality**:
   - Validate all input data before processing
   - Implement comprehensive error handling
   - Use consistent naming conventions

3. **Performance**:
   - Process files in batches for better throughput
   - Use appropriate parallelism levels
   - Monitor upload speeds and adjust configuration

4. **Monitoring**:
   - Enable comprehensive logging
   - Track success/failure rates
   - Monitor DataHub and storage system health

## Deployment Integration Details

### JuiceFS Configuration
Our deployed JuiceFS instance provides:
- **Distributed Storage**: High-performance file system across multiple nodes
- **S3 Compatibility**: Standard S3 API for seamless integration
- **Scalability**: Automatic scaling based on storage needs
- **Redundancy**: Built-in replication and fault tolerance

### DataHub Configuration
Our deployed DataHub instance offers:
- **Web UI**: User-friendly interface for data discovery
- **GraphQL API**: Programmatic access to metadata
- **Search Engine**: Elasticsearch-powered full-text search
- **Lineage Engine**: Track data transformations and dependencies

### Integration Points
1. **Metadata Sync**: This module populates DataHub with file metadata
2. **URL Linking**: Presigned URLs connect DataHub entries to JuiceFS files
3. **Browse Paths**: Hierarchical organization mirrors JuiceFS structure
4. **Tags & Properties**: Rich metadata enables powerful search and filtering

## Future Enhancements

- **Real-time Sync**: Automatic metadata updates when files change
- **Advanced Search**: ML-powered document similarity and recommendations  
- **API Gateway**: Unified API for both metadata and file access
- **Monitoring Dashboard**: Real-time system health and usage metrics
- **Data Quality Metrics**: Automated validation and quality scoring
- **Workflow Integration**: Direct connection to ML training pipelines
- Integration with additional storage backends
- Advanced retry mechanisms with exponential backoff
- Support for additional metadata standards and schemas
