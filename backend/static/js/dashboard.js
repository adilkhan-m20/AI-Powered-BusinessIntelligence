
// static/js/dashboard.js - Frontend Dashboard Logic

// API Configuration
const API_BASE_URL = window.location.origin;
const WS_URL = `ws://${window.location.host}`;

// Dashboard Application
class DashboardApp {
    constructor() {
        this.user = {
            id: 1,
            username: 'Demo User',
            token: 'demo_token'
        };
        
        this.stats = {
            totalDocuments: 0,
            processedDocuments: 0,
            totalQueries: 0,
            storageUsedMB: 0
        };
        
        this.documents = [];
        this.uploads = [];
        this.ws = null;
        this.isQuerying = false;
        
        // DOM Elements
        this.elements = {
            totalDocuments: document.getElementById('totalDocuments'),
            processedDocuments: document.getElementById('processedDocuments'),
            totalQueries: document.getElementById('totalQueries'),
            storageUsed: document.getElementById('storageUsed'),
            username: document.getElementById('username'),
            uploadProgress: document.getElementById('uploadProgress'),
            uploadList: document.getElementById('uploadList'),
            queryInput: document.getElementById('queryInput'),
            askBtn: document.getElementById('askBtn'),
            queryResult: document.getElementById('queryResult'),
            responseText: document.getElementById('responseText'),
            sourcesSection: document.getElementById('sourcesSection'),
            sourcesList: document.getElementById('sourcesList'),
            documentsTable: document.getElementById('documentsTable'),
            dropZone: document.getElementById('dropZone'),
            fileUpload: document.getElementById('file-upload'),
            logoutBtn: document.getElementById('logoutBtn'),
            notification: document.getElementById('notification'),
            notificationIcon: document.getElementById('notificationIcon'),
            notificationTitle: document.getElementById('notificationTitle'),
            notificationMessage: document.getElementById('notificationMessage')
        };
        
        // Bind methods
        this.init = this.init.bind(this);
        this.connectWebSocket = this.connectWebSocket.bind(this);
        this.handleWebSocketMessage = this.handleWebSocketMessage.bind(this);
        this.processFiles = this.processFiles.bind(this);
        this.uploadFile = this.uploadFile.bind(this);
        this.askQuestion = this.askQuestion.bind(this);
        this.deleteDocument = this.deleteDocument.bind(this);
        this.showNotification = this.showNotification.bind(this);
        
        // Event listeners
        this.elements.fileUpload.addEventListener('change', (e) => this.processFiles(Array.from(e.target.files)));
        this.elements.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.dropZone.classList.add('border-blue-500', 'bg-blue-50');
        });
        this.elements.dropZone.addEventListener('dragleave', () => {
            this.elements.dropZone.classList.remove('border-blue-500', 'bg-blue-50');
        });
        this.elements.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.dropZone.classList.remove('border-blue-500', 'bg-blue-50');
            this.processFiles(Array.from(e.dataTransfer.files));
        });
        this.elements.askBtn.addEventListener('click', this.askQuestion);
        this.elements.queryInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') this.askQuestion();
        });
        this.elements.logoutBtn.addEventListener('click', () => {
            localStorage.clear();
            location.reload();
        });
        
        // Initialize
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing EchoWeave Dashboard...');
        
        // Set username
        this.elements.username.textContent = this.user.username;
        
        try {
            await this.loadDashboard();
            this.connectWebSocket();
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showNotification('Error', 'Failed to load dashboard', 'error');
        }
    }
    
    async loadDashboard() {
        console.log('📊 Loading dashboard data...');
        
        try {
            // Load stats and documents
            await Promise.all([
                this.loadStats(),
                this.loadDocuments()
            ]);
            
            console.log('✅ Dashboard loaded successfully');
        } catch (error) {
            console.error('Dashboard loading error:', error);
            // Fallback to mock data
            this.loadMockData();
        }
    }
    
    async loadStats() {
        try {
            // In production, this would call: /analytics/dashboard
            await this.delay(500);
            
            this.stats = {
                totalDocuments: Math.floor(Math.random() * 20) + 5,
                processedDocuments: Math.floor(Math.random() * 15) + 3,
                totalQueries: Math.floor(Math.random() * 50) + 10,
                storageUsedMB: Math.random() * 500 + 50
            };
            
            // Update UI
            this.elements.totalDocuments.textContent = this.stats.totalDocuments;
            this.elements.processedDocuments.textContent = this.stats.processedDocuments;
            this.elements.totalQueries.textContent = this.stats.totalQueries;
            this.elements.storageUsed.textContent = `${this.stats.storageUsedMB.toFixed(1)} MB`;
            
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
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
            
            // Update UI
            this.updateDocumentsTable();
            
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }
    
    loadMockData() {
        console.log('📝 Loading mock data for demo...');
        this.stats = {
            totalDocuments: 12,
            processedDocuments: 8,
            totalQueries: 34,
            storageUsedMB: 245.7
        };
        
        // Update UI
        this.elements.totalDocuments.textContent = this.stats.totalDocuments;
        this.elements.processedDocuments.textContent = this.stats.processedDocuments;
        this.elements.totalQueries.textContent = this.stats.totalQueries;
        this.elements.storageUsed.textContent = `${this.stats.storageUsedMB.toFixed(1)} MB`;
        
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
                status: 'completed',
                file_size: 15360,
                created_at: new Date(Date.now() - 3600000).toISOString()
            },
            {
                id: 3,
                original_filename: 'financial_data.xlsx',
                status: 'completed',
                file_size: 5242880,
                created_at: new Date().toISOString()
            }
        ];
        
        this.updateDocumentsTable();
    }
    
    updateDocumentsTable() {
        this.elements.documentsTable.innerHTML = '';
        
        this.documents.forEach(doc => {
            const row = document.createElement('tr');
            
            // Status class mapping
            const statusClasses = {
                'completed': 'bg-green-100 text-green-800',
                'processing': 'bg-yellow-100 text-yellow-800',
                'pending': 'bg-gray-100 text-gray-800',
                'failed': 'bg-red-100 text-red-800'
            };
            
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                        <i class="fas fa-file mr-2 text-gray-400"></i>
                        <span>${doc.original_filename}</span>
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 py-1 text-xs rounded-full ${statusClasses[doc.status]}">
                        ${doc.status}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${this.formatFileSize(doc.file_size)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${this.formatDate(doc.created_at)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button data-id="${doc.id}" class="delete-doc text-red-600 hover:text-red-900">Delete</button>
                </td>
            `;
            
            this.elements.documentsTable.appendChild(row);
        });
        
        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-doc').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const docId = parseInt(e.target.dataset.id);
                this.deleteDocument(docId);
            });
        });
    }
    
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
                console.log('⚠️ WebSocket disconnected');
                setTimeout(() => this.connectWebSocket(), 5000); // Reconnect
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
        }
    }
    
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
    }
    
    handleTaskUpdate(data) {
        // Update upload progress
        const upload = this.uploads.find(u => u.taskId === data.task_id);
        if (upload) {
            upload.progress = data.progress;
            upload.status = data.status;
            
            // Update UI
            const uploadElement = document.getElementById(`upload-${upload.id}`);
            if (uploadElement) {
                const progressBar = uploadElement.querySelector('.progress-bar');
                const statusElement = uploadElement.querySelector('.status');
                
                progressBar.style.width = `${upload.progress}%`;
                statusElement.textContent = data.message || upload.status;
                
                if (data.status === 'completed') {
                    this.showNotification('Success', `${upload.filename} processed successfully`, 'success');
                    setTimeout(() => {
                        uploadElement.remove();
                        if (this.elements.uploadList.children.length === 0) {
                            this.elements.uploadProgress.classList.add('hidden');
                        }
                        this.loadDocuments(); // Refresh documents list
                    }, 2000);
                } else if (data.status === 'failed') {
                    statusElement.classList.add('text-red-500');
                }
            }
        }
    }
    
    handleDocumentUploaded(data) {
        this.showNotification('Uploaded', 'Document uploaded and queued for processing', 'success');
    }
    
    async processFiles(files) {
        console.log('📁 Processing files:', files.length);
        
        for (const file of files) {
            // Validate file
            if (!this.validateFile(file)) {
                continue;
            }
            
            // Create upload entry
            const uploadId = Date.now() + Math.random().toString(36).substr(2, 5);
            const upload = {
                id: uploadId,
                filename: file.name,
                status: 'uploading',
                progress: 0,
                taskId: null
            };
            
            this.uploads.push(upload);
            
            // Show upload progress
            this.elements.uploadProgress.classList.remove('hidden');
            
            // Add to UI
            const uploadElement = document.createElement('div');
            uploadElement.id = `upload-${upload.id}`;
            uploadElement.className = 'bg-gray-50 rounded-lg p-4 mb-2';
            uploadElement.innerHTML = `
                <div class="flex justify-between items-center mb-2">
                    <span class="font-medium">${file.name}</span>
                    <span class="text-sm text-gray-500 status">${upload.status}</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-blue-600 h-2 rounded-full progress-bar" style="width: ${upload.progress}%"></div>
                </div>
            `;
            
            this.elements.uploadList.appendChild(uploadElement);
            
            try {
                await this.uploadFile(file, upload);
            } catch (error) {
                console.error('Upload failed:', error);
                upload.status = 'failed';
                this.showNotification('Error', `Failed to upload ${file.name}`, 'error');
                
                const uploadElement = document.getElementById(`upload-${upload.id}`);
                if (uploadElement) {
                    const statusElement = uploadElement.querySelector('.status');
                    statusElement.textContent = 'Failed';
                    statusElement.classList.add('text-red-500');
                }
            }
        }
    }
    
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
    }
    
    async uploadFile(file, upload) {
        try {
            // Simulate file upload progress
            for (let i = 0; i <= 100; i += 10) {
                upload.progress = i;
                const uploadElement = document.getElementById(`upload-${upload.id}`);
                if (uploadElement) {
                    const progressBar = uploadElement.querySelector('.progress-bar');
                    progressBar.style.width = `${i}%`;
                }
                await this.delay(100);
            }
            
            // Simulate API response
            upload.status = 'processing';
            upload.taskId = 'task_' + Date.now();
            
            // Update UI
            const uploadElement = document.getElementById(`upload-${upload.id}`);
            if (uploadElement) {
                const statusElement = uploadElement.querySelector('.status');
                statusElement.textContent = 'Processing';
            }
            
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
    }
    
    async askQuestion() {
        const query = this.elements.queryInput.value.trim();
        if (!query || this.isQuerying) return;
        
        console.log('🤔 Asking question:', query);
        
        this.isQuerying = true;
        this.elements.askBtn.disabled = true;
        this.elements.askBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
        
        try {
            // Simulate API call delay
            await this.delay(2000);
            
            // Mock response
            const response = {
                query: query,
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
            
            // Update UI
            this.elements.responseText.textContent = response.response;
            
            if (response.sources && response.sources.length > 0) {
                this.elements.sourcesList.innerHTML = '';
                
                response.sources.forEach(source => {
                    const sourceElement = document.createElement('div');
                    sourceElement.className = 'bg-white rounded p-3 mb-2 border-l-4 border-blue-500';
                    sourceElement.innerHTML = `
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-medium">${source.document_name}</span>
                            <span class="text-sm text-gray-500">
                                Relevance: ${(source.relevance_score * 100).toFixed(0)}%
                            </span>
                        </div>
                        <p class="text-sm text-gray-600">${source.chunk_text}</p>
                    `;
                    this.elements.sourcesList.appendChild(sourceElement);
                });
                
                this.elements.sourcesSection.classList.remove('hidden');
            }
            
            this.elements.queryResult.classList.remove('hidden');
            this.showNotification('Success', 'Query processed successfully', 'success');
            
        } catch (error) {
            console.error('Query failed:', error);
            this.showNotification('Error', 'Failed to process query', 'error');
        } finally {
            this.isQuerying = false;
            this.elements.askBtn.disabled = false;
            this.elements.askBtn.innerHTML = '<i class="fas fa-paper-plane mr-2"></i>Ask';
            this.elements.queryInput.value = '';
        }
    }
    
    async deleteDocument(docId) {
        if (!confirm('Are you sure you want to delete this document?')) {
            return;
        }
        
        try {
            // Simulate API call
            await this.delay(500);
            
            this.documents = this.documents.filter(doc => doc.id !== docId);
            this.updateDocumentsTable();
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
    }
    
    showNotification(title, message, level = 'info') {
        // Set icon based on level
        let iconClass = '';
        switch (level) {
            case 'success':
                iconClass = 'fas fa-check-circle text-green-500';
                break;
            case 'error':
                iconClass = 'fas fa-exclamation-circle text-red-500';
                break;
            case 'warning':
                iconClass = 'fas fa-exclamation-triangle text-yellow-500';
                break;
            default:
                iconClass = 'fas fa-info-circle text-blue-500';
        }
        
        this.elements.notificationIcon.className = iconClass;
        this.elements.notificationTitle.textContent = title;
        this.elements.notificationMessage.textContent = message;
        
        // Show notification
        this.elements.notification.classList.remove('hidden');
        
        // Auto hide after 5 seconds
        setTimeout(() => {
            this.elements.notification.classList.add('hidden');
        }, 5000);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardApp();
});