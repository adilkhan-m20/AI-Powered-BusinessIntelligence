
# Uploads Directory

This directory stores uploaded documents for processing.

## 📁 Directory Structure

```
uploads/
├── .gitkeep                 # Keep this directory in git
├── temp/                    # Temporary files during upload
├── processed/               # Successfully processed files
├── failed/                  # Files that failed processing
└── README.md               # This file
```

## 🔧 Configuration

### File Storage Settings
- **Maximum file size**: 100MB per file
- **Supported formats**: PDF, TXT, DOCX, XLSX, CSV
- **Storage location**: Local filesystem (configurable)
- **Cleanup policy**: Files older than 30 days are archived

### Security Features
- **Path traversal protection**: Prevents ../ attacks
- **MIME type validation**: Checks actual file content
- **Filename sanitization**: Removes dangerous characters
- **Virus scanning**: Optional integration with ClamAV

## 📋 File Management

### Upload Process
1. File uploaded via API endpoint
2. Temporary storage in `temp/` directory
3. Validation and security checks
4. Move to main `uploads/` directory
5. Queue for AI processing
6. Move to `processed/` or `failed/` based on result

### File Organization
```
uploads/
├── user_123/               # User-specific directories
│   ├── document_456.pdf    # Original files with UUID names
│   ├── document_457.txt
│   └── chunks/             # Processed text chunks
│       ├── doc_456_chunk_1.txt
│       └── doc_456_chunk_2.txt
└── shared/                 # System-wide documents
```

### Metadata Storage
Each uploaded file has associated metadata:
```json
{
    "original_filename": "company_report.pdf",
    "uuid_filename": "550e8400-e29b-41d4-a716-446655440000.pdf",
    "user_id": 123,
    "upload_timestamp": "2025-08-11T00:24:00Z",
    "file_size": 2048576,
    "mime_type": "application/pdf",
    "processing_status": "completed",
    "chunks_created": 15,
    "embeddings_generated": 15
}
```

## 🛡️ Security Considerations

### Access Control
- Files are accessible only by their owners
- Admin users can access all files
- Anonymous access is blocked
- JWT tokens required for all operations

### File Validation
```python
def validate_uploaded_file(file):
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise ValidationError("File too large")
    
    # Check MIME type
    if file.content_type not in ALLOWED_TYPES:
        raise ValidationError("Unsupported file type")
    
    # Check for malicious content
    scan_result = virus_scanner.scan(file)
    if scan_result.is_infected:
        raise SecurityError("Malicious file detected")
```

## 🔄 Backup & Recovery

### Backup Strategy
- **Daily backups** to cloud storage (AWS S3/Google Cloud)
- **Incremental backups** for large files
- **Retention policy**: 90 days for user files
- **Point-in-time recovery** for critical documents

### Recovery Process
1. Identify missing/corrupted files
2. Restore from latest backup
3. Verify file integrity
4. Reprocess if necessary
5. Update database records

## 📊 Storage Monitoring

### Disk Usage Tracking
```python
def get_storage_stats():
    return {
        "total_files": count_files(),
        "total_size_mb": get_directory_size() / (1024 * 1024),
        "storage_used_percent": get_disk_usage_percent(),
        "files_by_type": count_files_by_extension(),
        "oldest_file": get_oldest_file_date(),
        "newest_file": get_newest_file_date()
    }
```

### Cleanup Automation
- **Scheduled cleanup** every night at 2 AM
- **Archive old files** after 30 days
- **Delete temp files** older than 24 hours
- **Compress processed files** older than 7 days

## 🚨 Error Handling

### Common Issues
- **Disk space full**: Alert admins, reject new uploads
- **Permission denied**: Check filesystem permissions
- **Corrupted files**: Move to quarantine directory
- **Processing timeout**: Retry with extended timeout

### Monitoring Alerts
```python
# Monitor disk space
if get_disk_usage_percent() > 85:
    send_alert("Low disk space in uploads directory")

# Monitor processing failures
if get_failed_processing_rate() > 10:
    send_alert("High processing failure rate detected")
```

## 🔧 Maintenance Tasks

### Daily Tasks
- Clean up temporary files
- Verify file integrity
- Update storage statistics
- Archive old files

### Weekly Tasks
- Run full backup
- Analyze storage patterns
- Update security scans
- Performance optimization

### Monthly Tasks
- Storage capacity planning
- Security audit
- Performance review
- Archive cleanup

## 📝 API Integration

### Upload Endpoint
```http
POST /documents/upload
Content-Type: multipart/form-data

{
    "file": [binary data],
    "metadata": {
        "description": "Optional file description",
        "tags": ["tag1", "tag2"]
    }
}
```

### File Access
```http
GET /documents/{document_id}/download
Authorization: Bearer <token>

Returns: File stream with appropriate headers
```

## 🎯 Best Practices

### For Developers
1. Always validate file uploads
2. Use UUID filenames to prevent conflicts
3. Implement proper error handling
4. Log all file operations
5. Monitor storage usage

### For Users
1. Use descriptive filenames
2. Keep file sizes reasonable
3. Use appropriate file formats
4. Organize files with meaningful names
5. Clean up old files regularly

This uploads directory is designed to handle enterprise-scale document processing while maintaining security and performance! 🚀