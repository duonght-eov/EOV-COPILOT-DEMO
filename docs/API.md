# SpeedMaint Intelligence - API Documentation

## Base URL

```
http://localhost:8080/api
```

---

## Authentication

### Register

```http
POST /auth/register
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123",
  "email": "admin@example.com",   // optional
  "role": "admin"                 // admin | manager | default
}
```

**Response:**
```json
{
  "id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "role": "admin",
  "is_active": true,
  "created_at": "2026-01-20T10:00:00Z"
}
```

### Login

```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=admin123
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

### Get Current User

```http
GET /auth/me
Authorization: Bearer <token>
```

---

## Workspaces

### List Workspaces

```http
GET /workspaces
Authorization: Bearer <token>
```

### Create Workspace

```http
POST /workspaces/new
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Workspace",
  "description": "Optional description"
}
```

### Get Workspace

```http
GET /workspace/{slug}
Authorization: Bearer <token>
```

### Update Workspace

```http
PUT /workspace/{slug}
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Updated Name",
  "query_mode": "hybrid"
}
```

### Delete Workspace

```http
DELETE /workspace/{slug}
Authorization: Bearer <token>
```

---

## Documents

### Upload Documents

```http
POST /workspace/{slug}/update-embeddings
Authorization: Bearer <token>
Content-Type: multipart/form-data

files: <file1>, <file2>, ...
```

**Response:**
```json
{
  "success": true,
  "message": "Uploaded 2 documents",
  "documents": [
    {"id": 1, "filename": "doc.pdf", "status": "pending"}
  ]
}
```

### List Documents

```http
GET /workspace/{slug}/documents
Authorization: Bearer <token>
```

### Delete Document

```http
DELETE /workspace/{slug}/documents/{doc_id}
Authorization: Bearer <token>
```

---

## Chat

### Send Message (Streaming)

```http
POST /workspace/{slug}/chat
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "What is this about?",
  "mode": "hybrid"    // local | global | hybrid | mix
}
```

**Response:** Server-Sent Events (SSE)
```
data: {"textResponse": "The document ", "close": false}
data: {"textResponse": "discusses...", "close": false}
data: {"textResponse": "", "close": true, "sources": []}
```

### Get Chat History

```http
GET /workspace/{slug}/chats?limit=50&offset=0
Authorization: Bearer <token>
```

### Clear Chat History

```http
DELETE /workspace/{slug}/chats
Authorization: Bearer <token>
```

---

## Admin - Users

### List Users (Admin only)

```http
GET /admin/users
Authorization: Bearer <admin_token>
```

### Create User

```http
POST /admin/users
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "username": "newuser",
  "password": "password123",
  "role": "default"
}
```

### Delete User

```http
DELETE /admin/users/{user_id}
Authorization: Bearer <admin_token>
```

---

## System

### Health Check

```http
GET /health
```

### System Info

```http
GET /system/
```

### LLM Configuration

```http
GET /system/llm-preference
Authorization: Bearer <token>
```

---

## Error Responses

```json
{
  "detail": "Error message"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Error |
