
// static/js/dashboard.js - Frontend Dashboard Logic

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

// Dashboard Application
function dashboardApp() {
    return {
        // State
        user: {
            id: null,
            username: 'Demo User',
            token: localStorage.getItem('auth_token') || null
        },
        
        stats: {
            totalDocuments: 0,
            processedDocuments: 0,
            totalQueries: 0,
            storageUsedMB: 0
        },
        
        documents: [],
        uploads: [],
        query: '',
        queryResult: null,
        notification: null,
        isQuerying: false,
        ws: null,
        
        // Initialize
        async init() {
            console.log('🚀 Initializing MultiModal AI Dashboard...');
            
            // Check authentication
            if (!this.user.token) {
                this.showLogin();
                return;
            }
            
            try {
                await this.loadDashboard();
                this.connectWebSocket();
            } catch (error) {
                console.error('Failed to initialize dashboard:', error);
                this.showNotification('Error', 'Failed to load dashboard', 'error');
            }
        },
        
        // Authentication
        showLogin() {
            // For demo purposes, create a demo token
            this.user.token = 'demo_token_' + Date.now();
            this.user.id = 1;
            this.user.username = 'Demo User';
            localStorage.setItem('auth_token', this.user.token);
            this.loadDashboard();
        },
        
        logout() {
            localStorage.removeItem('auth_token');
            this.user.token = null;
            if (this.ws) {
                this.ws.close();
            }
            location.reload();
        },
        
        // Dashboard Data Loading
        async loadDashboard() {
            console.log('📊 Loading dashboard data...');
            
            try {
                // Load stats, documents, etc.
                await Promise.all([
                    this.loadStats(),
                    this.loadDocuments()
                ]);
                
                console.log('✅ Dashboard loaded successfully');
            } catch (error) {
                console.error('Dashboard loading error:', error);
                // For demo, use mock data
                this.loadMockData();
            }
        },
        
        async loadStats() {
            try {
                // In production, this would call: /analytics/dashboard
                // For demo, simulate API call
                await this.delay(500);
                
                this.stats = {
                    totalDocuments: Math.floor(Math.random() * 20) + 5,
                    processedDocuments: Math.floor(Math.random() * 15) + 3,
                    totalQueries: Math.floor(Math.random() * 50) + 10,
                    storageUsedMB: Math.random() * 500 + 50
                };
                
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        },
        
        async loadDocuments() {
            try {
                // In production: /documents
                await this.delay(300);
                
                this.documents = [
                    {
                        id: 1,
                        original_filename: 'company_report.pdf',
                        status: 'completed',
                        file_size: 2048576,
                        created_at: new Date(Date.now() - 86400000).toISOString()
                    },
                    {
                        id: 2,
                        original_filename: 'meeting_notes.txt',
                        status: 'processing',
                        file_size: 15360,
                        created_at: new Date(Date.now() - 3600000).toISOString()
                    },
                    {
                        id: 3,
                        original_filename: 'financial_data.xlsx',
                        status: 'pending',
                        file_size: 5242880,
                        created_at: new Date().toISOString()
                    }
                ];
                
            } catch (error) {
                console.error('Failed to load documents:', error);
            }
        },
        
        loadMockData() {
            console.log('📝 Loading mock data for demo...');
            this.stats = {
                totalDocuments: 12,
                processedDocuments: 8,
                totalQueries: 34,
                storageUsedMB: 245.7
            };
        },
        
        // WebSocket Connection
        connectWebSocket() {
            if (!this.user.id) return;
            
            try {
                const wsUrl = `${WS_URL}/ws/${this.user.id}`;
                console.log('🔌 Connecting to WebSocket:', wsUrl);
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('✅ WebSocket connected');
                    this.showNotification('Connected', 'Real-time updates enabled', 'success');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('❌ WebSocket disconnected');
                    setTimeout(() => this.connectWebSocket(), 5000); // Reconnect
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
                
            } catch (error) {
                console.error('WebSocket connection failed:', error);
            }
        },
        
        handleWebSocketMessage(data) {
            console.log('📨 WebSocket message:', data);
            
            switch (data.type) {
                case 'task_update':
                    this.handleTaskUpdate(data);
                    break;
                case 'document_uploaded':
                    this.handleDocumentUploaded(data);
                    break;
                case 'system_message':
                    this.showNotification('System', data.message, data.level);
                    break;
                default:
                    console.log('Unknown WebSocket message type:', data.type);
            }
        },
        
        handleTaskUpdate(data) {
            // Update upload progress
            const upload = this.uploads.find(u => u.taskId === data.task_id);
            if (upload) {
                upload.progress = data.progress;
                upload.status = data.status;
                
                if (data.status === 'completed') {
                    this.showNotification('Success', `${upload.filename} processed successfully`, 'success');
                    setTimeout(() => {
                        this.uploads = this.uploads.filter(u => u.taskId !== data.task_id);
                        this.loadDocuments(); // Refresh documents list
                    }, 2000);
                }
            }
        },
        
        handleDocumentUploaded(data) {
            this.showNotification('Uploaded', 'Document uploaded and queued for processing', 'success');
        },
        
        // File Upload Handling
        handleFileSelect(event) {
            const files = Array.from(event.target.files);
            this.processFiles(files);
        },
        
        handleDrop(event) {
            const files = Array.from(event.dataTransfer.files);
            this.processFiles(files);
        },
        
        async processFiles(files) {
            console.log('📁 Processing files:', files.length);
            
            for (const file of files) {
                // Validate file
                if (!this.validateFile(file)) {
                    continue;
                }
                
                // Create upload entry
                const uploadId = Date.now() + Math.random();
                const upload = {
                    id: uploadId,
                    filename: file.name,
                    status: 'uploading',
                    progress: 0,
                    taskId: null
                };
                
                this.uploads.push(upload);
                
                try {
                    await this.uploadFile(file, upload);
                } catch (error) {
                    console.error('Upload failed:', error);
                    upload.status = 'failed';
                    this.showNotification('Error', `Failed to upload ${file.name}`, 'error');
                }
            }
        },
        
        validateFile(file) {
            const maxSize = 100 * 1024 * 1024; // 100MB
            const allowedTypes = ['.pdf', '.txt', '.docx', '.xlsx', '.csv'];
            
            // Check size
            if (file.size > maxSize) {
                this.showNotification('Error', `File too large: ${file.name} (max 100MB)`, 'error');
                return false;
            }
            
            // Check type
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedTypes.includes(extension)) {
                this.showNotification('Error', `Unsupported file type: ${extension}`, 'error');
                return false;
            }
            
            return true;
        },
        
        async uploadFile(file, upload) {
            try {
                // Simulate file upload progress
                for (let i = 0; i <= 100; i += 10) {
                    upload.progress = i;
                    await this.delay(100);
                }
                
                // Simulate API response
                upload.status = 'processing';
                upload.taskId = 'task_' + Date.now();
                
                // In production, this would be:
                /*
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/documents/upload', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.user.token}`
                    },
                    body: formData
                });
                
                const result = await response.json();
                upload.taskId = result.task_id;
                */
                
            } catch (error) {
                upload.status = 'failed';
                throw error;
            }
        },
        
        // RAG Query Handling
        async askQuestion() {
            if (!this.query.trim() || this.isQuerying) return;
            
            console.log('🤔 Asking question:', this.query);
            
            this.isQuerying = true;
            this.queryResult = null;
            
            try {
                // Simulate API call delay
                await this.delay(2000);
                
                // Mock response
                this.queryResult = {
                    query: this.query,
                    response: "Based on your documents, I can provide you with the following information. This is a simulated response that demonstrates how the RAG system would work with your actual documents.",
                    sources: [
                        {
                            document_id: 1,
                            document_name: "company_report.pdf",
                            chunk_text: "This section contains relevant information about your query...",
                            relevance_score: 0.89
                        },
                        {
                            document_id: 2,
                            document_name: "meeting_notes.txt",
                            chunk_text: "Additional context from meeting notes that relates to your question...",
                            relevance_score: 0.76
                        }
                    ],
                    confidence: 0.85,
                    processing_time: 1.234
                };
                
                // In production:
                /*
                const response = await fetch('/rag/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.user.token}`
                    },
                    body: JSON.stringify({
                        query: this.query,
                        max_sources: 5
                    })
                });
                
                this.queryResult = await response.json();
                */
                
            } catch (error) {
                console.error('Query failed:', error);
                this.showNotification('Error', 'Failed to process query', 'error');
            } finally {
                this.isQuerying = false;
            }
        },
        
        // Document Management
        async deleteDocument(docId) {
            if (!confirm('Are you sure you want to delete this document?')) {
                return;
            }
            
            try {
                // Simulate API call
                await this.delay(500);
                
                this.documents = this.documents.filter(doc => doc.id !== docId);
                this.showNotification('Success', 'Document deleted', 'success');
                
                // In production:
                /*
                await fetch(`/documents/${docId}`, {
                    method: 'DELETE',
                    headers: {
                        'Authorization': `Bearer ${this.user.token}`
                    }
                });
                */
                
            } catch (error) {
                console.error('Delete failed:', error);
                this.showNotification('Error', 'Failed to delete document', 'error');
            }
        },
        
        // Utility Functions
        showNotification(title, message, level = 'info') {
            this.notification = { title, message, level };
            
            setTimeout(() => {
                this.notification = null;
            }, 5000);
        },
        
        formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        
        formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        },
        
        delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
    };
}