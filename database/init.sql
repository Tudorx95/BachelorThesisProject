-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create files table
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create simulation_results table (for storing FL simulation results)
CREATE TABLE IF NOT EXISTS simulation_results (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
    task_id VARCHAR(100) UNIQUE NOT NULL,
    simulation_config JSONB NOT NULL,
    results JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_files_project_id ON files(project_id);
CREATE INDEX idx_simulation_results_user_id ON simulation_results(user_id);
CREATE INDEX idx_simulation_results_task_id ON simulation_results(task_id);
CREATE INDEX idx_simulation_results_status ON simulation_results(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at BEFORE UPDATE ON files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert demo user (password: "demo123" - bcrypt hashed)
-- Hash generated with: from passlib.context import CryptContext; pwd_context = CryptContext(schemes=["bcrypt"]); pwd_context.hash("demo123")
INSERT INTO users (username, email, password_hash) VALUES 
('demo', 'demo@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5lWf.3YpqJ3IG')
ON CONFLICT (username) DO NOTHING;

-- Insert demo project
INSERT INTO projects (user_id, name, description) VALUES 
(1, 'FL Simulation Demo', 'Demo project for federated learning simulation')
ON CONFLICT DO NOTHING;

-- Insert demo file
INSERT INTO files (project_id, name, content) VALUES 
(1, 'Welcome.md', '# Welcome to FL Simulator!

## Getting Started

To run a federated learning simulation, follow these steps:

### 1. Configure Simulation Parameters
Click on the "Simulation Options" button to configure:
- Number of clients (N)
- Number of malicious clients (M)
- Neural network model name
- Training rounds
- Data poisoning strategy

### 2. Add Your Training Code
Write or paste your PyTorch/TensorFlow model training code in the code editor.

### 3. Run Simulation
Click the "Run" button to start the simulation. The system will:
1. Execute your code
2. Generate clean data
3. Apply data poisoning (if enabled)
4. Run FL simulation across multiple clients
5. Generate comprehensive results

### Example Code Structure

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Your training code here
model = SimpleNN()
# ... training logic
```

## Features

✅ Multi-client FL simulation
✅ Data poisoning attack simulation
✅ Real-time progress tracking
✅ Result persistence and analysis
✅ Multiple simulation strategies

Happy simulating! 🚀
')
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "user";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "user";