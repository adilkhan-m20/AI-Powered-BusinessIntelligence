
# Static Files Documentation

## 📁 Frontend Dashboard Components

This directory contains the frontend assets for the MultiModal AI application dashboard.

### 🎯 Files Overview

#### `index.html` [95]
- **Complete dashboard interface** with modern UI
- **Real-time file upload** with drag & drop
- **Live document processing** status updates
- **RAG query interface** with source citations
- **WebSocket integration** for real-time notifications
- **Responsive design** using Tailwind CSS

#### `js/dashboard.js` [96]
- **Frontend application logic** using Alpine.js
- **WebSocket connection management**
- **File upload handling** with progress tracking
- **RAG query processing**
- **Real-time notifications**
- **Demo mode** with mock data for testing

#### `css/style.css` [97]
- **Custom styling** and animations
- **Progress bars** and status indicators
- **Notification styles**
- **Responsive design** utilities
- **Dark mode support**
- **Loading states** and transitions

### 🚀 Features

#### ✅ **File Management**
- Upload multiple files via drag & drop or file picker
- Real-time upload progress tracking
- File validation (size, type)
- Document status monitoring (pending, processing, completed)

#### ✅ **RAG Interface**
- Ask questions about uploaded documents
- View response with source citations
- Real-time query processing feedback
- Source relevance scoring

#### ✅ **Real-time Updates**
- WebSocket connection for live updates
- Processing status notifications
- System alerts and messages
- Auto-reconnection on disconnect

#### ✅ **Analytics Dashboard**
- Document statistics
- Storage usage tracking
- Query performance metrics
- User activity overview

### 🛠 Integration with Backend

The frontend automatically connects to:
- **REST API**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:8000/ws/{user_id}`
- **File Upload**: `/documents/upload`
- **RAG Queries**: `/rag/query`

### 🎨 UI Components

#### Dashboard Cards
```html
<div class="bg-white rounded-lg shadow p-6">
    <div class="flex items-center">
        <div class="p-2 bg-blue-100 rounded-lg">
            <i class="fas fa-file text-blue-600"></i>
        </div>
        <div class="ml-4">
            <p class="text-sm text-gray-600">Total Documents</p>
            <p class="text-2xl font-bold">12</p>
        </div>
    </div>
</div>
```

#### File Upload Zone
```html
<div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center"
     @dragover.prevent @drop.prevent="handleDrop($event)">
    <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-4"></i>
    <p class="text-lg text-gray-600 mb-2">Drop files here or</p>
    <label class="bg-blue-500 text-white px-6 py-3 rounded cursor-pointer">
        Choose Files
    </label>
</div>
```

### 📱 Responsive Design

The dashboard is fully responsive with:
- **Mobile-first approach** using Tailwind CSS
- **Breakpoint-specific layouts** for tablets and desktops
- **Touch-friendly interactions** for mobile devices
- **Collapsible navigation** on smaller screens

### 🔧 Customization

#### Colors & Themes
Modify CSS variables in `style.css`:
```css
:root {
    --primary-color: #3b82f6;
    --secondary-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
}
```

#### API Endpoints
Update endpoints in `dashboard.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';
```

### 🧪 Demo Mode

The dashboard includes a **demo mode** that:
- Works without backend connection
- Shows mock data and animations
- Demonstrates all UI features
- Perfect for presentations and testing

### 🔐 Authentication

Currently uses a **demo token system** for development:
```javascript
// Demo authentication
this.user.token = 'demo_token_' + Date.now();
```

For production, integrate with your JWT authentication system.

### 📊 Real-time Features

#### WebSocket Message Types
- `task_update`: Document processing progress
- `document_uploaded`: Upload completion
- `system_message`: System notifications
- `connection_established`: WebSocket connected

#### Progress Tracking
```javascript
// Upload progress tracking
const upload = {
    id: uploadId,
    filename: file.name,
    status: 'uploading',  // uploading, processing, completed, failed
    progress: 0,          // 0-100
    taskId: null
};
```

### 🎯 Production Deployment

For production deployment:

1. **Replace demo authentication** with real JWT integration
2. **Update API endpoints** to production URLs
3. **Enable HTTPS** for WebSocket connections
4. **Add error handling** for network failures
5. **Implement user management** features

### 🚀 Getting Started

1. **Place files in backend/static/**:
   ```
   backend/static/
   ├── index.html
   ├── js/
   │   └── dashboard.js
   └── css/
       └── style.css
   ```

2. **Access the dashboard**:
   ```
   http://localhost:8000/static/index.html
   ```

3. **Test features**:
   - Upload files via drag & drop
   - Ask questions in the RAG interface
   - Monitor real-time updates

The dashboard provides a **professional, enterprise-grade interface** that showcases your MultiModal AI system beautifully! 🎉