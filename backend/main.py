from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import jwt
import json
import asyncio
import os
import uuid
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper to read Docker secrets from file
def read_secret(env_file_var: str, fallback_env: str = None, default: str = "") -> str:
    """Read a secret from a Docker secret file, with fallback to env var and default."""
    secret_file = os.getenv(env_file_var)
    if secret_file and os.path.isfile(secret_file):
        with open(secret_file, "r") as f:
            return f.read().strip()
    if fallback_env:
        return os.getenv(fallback_env, default)
    return default

# Database configuration
_db_password = read_secret("DB_PASSWORD_FILE", fallback_env="DB_PASSWORD", default="pass")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://user:{_db_password}@localhost:5432/fl_db")
SECRET_KEY = read_secret("SECRET_KEY_FILE", fallback_env="SECRET_KEY", default="your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour session

# Orchestrator configuration
ORCHESTRATOR_URL = "http://10.13.0.102:8000"  # IP-ul serverului ATM
ORCHESTRATOR_USER = "tudor"
ORCHESTRATOR_PASSWORD = read_secret("ORCHESTRATOR_PASSWORD_FILE", fallback_env="ORCHESTRATOR_PASSWORD", default="")

# Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="FL Simulator API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000", "http://10.13.170.3:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dict to track active tasks
active_tasks: Dict[str, Dict] = {}

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="projects")
    files = relationship("File", back_populates="project", cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="files")
    simulation_results = relationship("SimulationResult", back_populates="file", cascade="all, delete-orphan")

class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    task_id = Column(String, unique=True, nullable=False, index=True)
    simulation_config = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    output = Column(Text, nullable=True)  # Store console output
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    file = relationship("File", back_populates="simulation_results")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class FileCreate(BaseModel):
    name: str
    content: str

class FileUpdate(BaseModel):
    content: str

class FileRename(BaseModel):
    name: str

class FileReorder(BaseModel):
    file_id: int
    new_order: int

class FilesReorderBulk(BaseModel):
    updates: List[FileReorder]

class FileMoveProject(BaseModel):
    new_project_id: int
    new_order: Optional[int] = None

class FileResponse(BaseModel):
    id: int
    project_id: int
    name: str
    content: str
    order: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SimulationConfig(BaseModel):
    N: int
    M: int
    NN_NAME: str
    R: int
    ROUNDS: int
    strategy: str = "first"
    poison_operation: str = "noise"
    poison_intensity: float = 0.1
    poison_percentage: float = 0.2
    data_poison_protection: str = "fedavg"

class RunRequest(BaseModel):
    filename: str
    code: str
    simulation_config: Optional[SimulationConfig] = None

class SimulationResultCreate(BaseModel):
    file_id: int
    project_id: int
    task_id: str
    simulation_config: dict
    results: Optional[dict] = None
    output: Optional[str] = None
    status: str = "pending"

class SimulationResultResponse(BaseModel):
    id: int
    user_id: int
    project_id: Optional[int]
    file_id: Optional[int]
    task_id: str
    simulation_config: dict
    results: Optional[dict]
    output: Optional[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# Orchestrator communication functions
def login_to_orchestrator():
    """Login to the orchestrator server and get token"""
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/login",
            json={"username": ORCHESTRATOR_USER, "password": ORCHESTRATOR_PASSWORD},
            timeout=10
        )
        if response.status_code == 200:
            token = response.json().get("token")
            logger.info("Successfully logged in to orchestrator")
            return token
        else:
            logger.error(f"Failed to login to orchestrator: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to orchestrator: {str(e)}")
        return None

def send_simulation_to_orchestrator(task_id: str, template_code: str, config: SimulationConfig, user_id: int):
    """Send simulation request to orchestrator"""
    token = login_to_orchestrator()
    if not token:
        raise HTTPException(status_code=503, detail="Cannot connect to orchestrator server")
    
    # Prepare simulation command
    simulation_data = {
        "task_id": task_id,
        "user_id": user_id,
        "template_code": template_code,
        "config": {
            "N": config.N,
            "M": config.M,
            "NN_NAME": config.NN_NAME,
            "R": config.R,
            "ROUNDS": config.ROUNDS,
            "strategy": config.strategy,
            "poison_operation": config.poison_operation,
            "poison_intensity": config.poison_intensity,
            "poison_percentage": config.poison_percentage,
            "data_poison_protection": config.data_poison_protection
        }
    }
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        logger.info("Before sending simulation to orchestrator")
        response = requests.post(
            f"{ORCHESTRATOR_URL}/simulate",
            json=simulation_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info(f"Simulation task {task_id} sent to orchestrator successfully")
            return response.json()
        else:
            logger.error(f"Orchestrator returned error: {response.text}")
            raise HTTPException(status_code=500, detail=f"Orchestrator error: {response.text}")
    except requests.exceptions.Timeout:
        logger.error("Orchestrator request timed out")
        raise HTTPException(status_code=504, detail="Orchestrator request timed out")
    except Exception as e:
        logger.error(f"Error sending to orchestrator: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def cancel_simulation_on_orchestrator(task_id: str, user_id: int):
    """Send cancellation request to orchestrator to stop a running simulation"""
    token = login_to_orchestrator()
    if not token:
        raise HTTPException(status_code=503, detail="Cannot connect to orchestrator server")
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"Sending cancellation request for task {task_id} to orchestrator")
        
        response = requests.post(
            f"{ORCHESTRATOR_URL}/cancel/{task_id}",
            json={"user_id": user_id},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info(f"Task {task_id} cancelled successfully on orchestrator")
            return response.json()
        elif response.status_code == 404:
            logger.warning(f"Task {task_id} not found on orchestrator")
            raise HTTPException(status_code=404, detail="Task not found on orchestrator")
        else:
            logger.error(f"Orchestrator returned error during cancellation: {response.text}")
            raise HTTPException(status_code=500, detail=f"Orchestrator cancellation error: {response.text}")
            
    except requests.exceptions.Timeout:
        logger.error("Orchestrator cancellation request timed out")
        raise HTTPException(status_code=504, detail="Orchestrator cancellation request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling on orchestrator: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cancellation error: {str(e)}")


async def poll_orchestrator_status(task_id: str):
    """Poll orchestrator for task status updates"""
    token = login_to_orchestrator()
    if not token:
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    while task_id in active_tasks:
        try:
            response = requests.get(
                f"{ORCHESTRATOR_URL}/status/{task_id}",
                headers=headers,
                timeout=10
            )
            
            if response.ok:
                status_data = response.json()
                active_tasks[task_id]["status"] = status_data
                
                if status_data.get("status") in ["completed", "error", "cancelled"]:
                    if status_data.get("status") == "completed":
                        # Fetch results only for completed tasks
                        results_response = requests.get(f"{ORCHESTRATOR_URL}/results/{task_id}", headers=headers)
                        if results_response.ok:
                            results_data = results_response.json()
                            # Include the simulation config in the results
                            if "config" in active_tasks[task_id]:
                                results_data["config"] = active_tasks[task_id]["config"]
                            active_tasks[task_id]["results_data"] = results_data
                            logger.info(f"Results with config for task {task_id}: {results_data}")
                    break
            
            await asyncio.sleep(5)  # Poll every 5 seconds
        except Exception as e:
            logger.error(f"Error polling orchestrator: {str(e)}")
            await asyncio.sleep(5)

# Authentication endpoints
@app.post("/api/auth/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username exists
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.id}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.post("/api/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.id}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Project endpoints
@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()
    return projects

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_project = Project(
        user_id=current_user.id,
        name=project.name,
        description=project.description
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@app.delete("/api/projects/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    return {"message": "Project deleted successfully"}

# File endpoints
@app.get("/api/projects/{project_id}/files", response_model=List[FileResponse])
async def get_files(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    files = db.query(File).filter(File.project_id == project_id).order_by(File.order).all()
    return files

@app.post("/api/projects/{project_id}/files", response_model=FileResponse)
async def create_file(
    project_id: int,
    file: FileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get max order for this project
    max_order = db.query(File).filter(File.project_id == project_id).count()

    db_file = File(
        project_id=project_id,
        name=file.name,
        content=file.content,
        order=max_order
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

@app.get("/api/files/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file = db.query(File).join(Project).filter(
        File.id == file_id,
        Project.user_id == current_user.id
    ).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@app.put("/api/files/{file_id}", response_model=FileResponse)
async def update_file(
    file_id: int,
    file_update: FileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file = db.query(File).join(Project).filter(
        File.id == file_id,
        Project.user_id == current_user.id
    ).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    file.content = file_update.content
    file.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(file)
    return file

@app.delete("/api/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file = db.query(File).join(Project).filter(
        File.id == file_id,
        Project.user_id == current_user.id
    ).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    db.delete(file)
    db.commit()
    return {"message": "File deleted successfully"}

# NEW: File rename endpoint
@app.patch("/api/files/{file_id}/rename", response_model=FileResponse)
async def rename_file(
    file_id: int,
    rename_data: FileRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rename a file"""
    file = db.query(File).join(Project).filter(
        File.id == file_id,
        Project.user_id == current_user.id
    ).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    file.name = rename_data.name
    file.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(file)
    return file

# NEW: File reorder endpoint (bulk update)
@app.post("/api/files/reorder")
async def reorder_files(
    reorder_data: FilesReorderBulk,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reorder multiple files at once"""
    try:
        for update in reorder_data.updates:
            file = db.query(File).join(Project).filter(
                File.id == update.file_id,
                Project.user_id == current_user.id
            ).first()
            if file:
                file.order = update.new_order
                file.updated_at = datetime.utcnow()

        db.commit()
        return {"message": "Files reordered successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error reordering files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reorder files: {str(e)}")

# NEW: Move file to different project
@app.patch("/api/files/{file_id}/move", response_model=FileResponse)
async def move_file_to_project(
    file_id: int,
    move_data: FileMoveProject,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Move a file from one project to another"""
    try:
        # Verify file exists and user owns it
        file = db.query(File).join(Project).filter(
            File.id == file_id,
            Project.user_id == current_user.id
        ).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # Verify new project exists and user owns it
        new_project = db.query(Project).filter(
            Project.id == move_data.new_project_id,
            Project.user_id == current_user.id
        ).first()
        if not new_project:
            raise HTTPException(status_code=404, detail="Target project not found")

        # Get order for new project
        if move_data.new_order is not None:
            new_order = move_data.new_order
        else:
            # Put at the end of the new project
            new_order = db.query(File).filter(File.project_id == move_data.new_project_id).count()

        # Update file
        file.project_id = move_data.new_project_id
        file.order = new_order
        file.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(file)
        return file
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error moving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to move file: {str(e)}")

# Simulation endpoint
@app.post("/run")
async def run_code(request: RunRequest, background_tasks: BackgroundTasks):
    # Generate unique task ID
    logger.info("Inside run endpoint")
    task_id = str(uuid.uuid4())
    # Initial output
    output = f"🚀 Starting FL Simulation...\n"
    output += f"Task ID: {task_id}\n"
    output += f"File: {request.filename}\n\n"
    

    logger.info("Prepare resquest simulation config")
    logger.info(f"{request.simulation_config}")

    
    if request.simulation_config:
        config = request.simulation_config
        output += f"📊 Configuration:\n"
        output += f"  • Clients: {config.N} (Malicious: {config.M})\n"
        output += f"  • Rounds: {config.ROUNDS} (Poisoned: {config.R})\n"
        output += f"  • Model: {config.NN_NAME}\n"
        output += f"  • Strategy: {config.strategy}\n\n"
        output += f"🦠 Data Poisoning Attack:\n"
        output += f"  • Operation: {config.poison_operation}\n"
        output += f"  • Intensity: {config.poison_intensity}\n"
        output += f"  • Percentage: {config.poison_percentage}\n\n"
        output += f"🛡️ Data Poison Protection:\n"
        output += f"  • Aggregation Method: {config.data_poison_protection}\n\n"
        
        # Store task info
        active_tasks[task_id] = {
            "status": "initializing",
            "config": config.dict(),
            "code": request.code
        }
        
        try:
            # Send to orchestrator
            output += "📤 Sending simulation to orchestrator server...\n"
            logger.info(f"Before Sending task {task_id} to orchestrator")
            result = send_simulation_to_orchestrator(
                task_id=task_id,
                template_code=request.code,
                config=config,
                user_id=1  # TODO: Get from auth
            )
            output += f"✅ Simulation queued on orchestrator\n"
            logger.info(f"After Sending Task {task_id}: response {result}")
            # Start background polling
            background_tasks.add_task(poll_orchestrator_status, task_id)
            
        except HTTPException as e:
            output += f"\n❌ Error: {e.detail}\n"
            return {
                "status": "error",
                "output": output,
                "task_id": None
            }
    else:
        output += "⚠️  No simulation config provided\n"
    
    return {
        "status": "success",
        "output": output,
        "task_id": task_id
    }

# NEW: Cancellation endpoint
@app.post("/cancel/{task_id}")
async def cancel_simulation(task_id: str):
    """Cancel a running simulation by forwarding the request to the orchestrator"""
    logger.info(f"Received cancellation request for task {task_id}")
    
    # Check if task exists in our tracking
    if task_id not in active_tasks:
        logger.warning(f"Task {task_id} not found in active tasks")
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    # Check if task is already in a terminal state
    if isinstance(task_info.get("status"), dict):
        current_status = task_info["status"].get("status")
        if current_status in ["completed", "cancelled", "error"]:
            logger.info(f"Task {task_id} already in terminal state: {current_status}")
            return {
                "status": "success",
                "message": f"Task already {current_status}",
                "task_id": task_id
            }
    
    try:
        # Forward cancellation request to orchestrator
        # TODO: Get actual user_id from authentication
        user_id = 1
        result = cancel_simulation_on_orchestrator(task_id, user_id)
        
        # Update local task status to reflect cancellation
        active_tasks[task_id]["status"] = {
            "status": "cancelled",
            "message": "Simulation cancelled by user",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully cancelled task {task_id}")
        
        return {
            "status": "success",
            "message": "Simulation cancelled successfully",
            "task_id": task_id,
            "details": result
        }
        
    except HTTPException as e:
        # Re-raise HTTP exceptions from orchestrator communication
        raise e
    except Exception as e:
        error_msg = f"Error cancelling task {task_id}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)



@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to simulation monitor"
        })
        logger.info(f"WebSocket client connected for task {task_id}")
        
        # Monitor task status
        while task_id in active_tasks:
            task_data = active_tasks[task_id]
            status_info = task_data.get("status", {})
            logger.info(f"WebSocket sending status for task {task_id}: {status_info}")
            
            if isinstance(status_info, dict):
                # Prepare the data to send
                data_to_send = dict(status_info)  # Create a copy
                
                # If task is completed and has results_data, include it
                if status_info.get("status") == "completed" and "results_data" in task_data:
                    data_to_send["results_data"] = task_data["results_data"]
                    logger.info(f"Including results_data for task {task_id}")
                
                await websocket.send_json({
                    "type": "orchestrator_update",
                    "data": data_to_send
                })
                
                # Check if completed, error, or cancelled
                if status_info.get("status") in ["completed", "error", "cancelled"]:
                    logger.info(f"Task {task_id} reached terminal state: {status_info.get('status')}")
                    # Clean up after a short delay to ensure client receives the final message
                    await asyncio.sleep(1)
                    del active_tasks[task_id]
                    break
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from task {task_id}")
        if task_id in active_tasks:
            del active_tasks[task_id]
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the current status of a task"""
    logger.info(f"Checking status for task {task_id}")

    # Check if task exists in our active tasks
    if task_id not in active_tasks:
        logger.warning(f"Task {task_id} not found in active tasks")
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = active_tasks[task_id]
    status_info = task_info.get("status", {})

    response = {
        "task_id": task_id,
        "status": status_info.get("status", "running") if isinstance(status_info, dict) else "running",
        "current_step": status_info.get("step") if isinstance(status_info, dict) else None,
        "message": status_info.get("message") if isinstance(status_info, dict) else None,
        "orchestrator_status": status_info if isinstance(status_info, dict) else None,
    }

    # Include results if completed
    if status_info.get("status") == "completed" and "results_data" in task_info:
        response["results"] = task_info["results_data"]

    logger.info(f"Task {task_id} status: {response}")
    return response


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "orchestrator_url": ORCHESTRATOR_URL
    }


# ==================== SIMULATION RESULTS ENDPOINTS ====================

@app.post("/api/simulation-results", response_model=SimulationResultResponse)
async def save_simulation_result(
    result_data: SimulationResultCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save simulation results to database"""
    try:
        # Check if simulation result already exists
        existing = db.query(SimulationResult).filter(
            SimulationResult.task_id == result_data.task_id
        ).first()
        
        if existing:
            # Update existing result
            existing.results = result_data.results
            existing.output = result_data.output
            existing.status = result_data.status
            if result_data.status == "completed":
                existing.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing

        # Create new simulation result
        sim_result = SimulationResult(
            user_id=current_user.id,
            project_id=result_data.project_id,
            file_id=result_data.file_id,
            task_id=result_data.task_id,
            simulation_config=result_data.simulation_config,
            results=result_data.results,
            output=result_data.output,
            status=result_data.status,
            completed_at=datetime.utcnow() if result_data.status == "completed" else None
        )
        
        db.add(sim_result)
        db.commit()
        db.refresh(sim_result)
        
        logger.info(f"Saved simulation result for task {result_data.task_id}")
        return sim_result
        
    except Exception as e:
        logger.error(f"Error saving simulation result: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save simulation result: {str(e)}")


@app.get("/api/files/{file_id}/simulation-results", response_model=Optional[SimulationResultResponse])
async def get_file_simulation_results(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the latest simulation results for a file (including running simulations)"""
    try:
        # First check for running simulations
        running_result = db.query(SimulationResult).filter(
            SimulationResult.file_id == file_id,
            SimulationResult.user_id == current_user.id,
            SimulationResult.status == "running"
        ).order_by(SimulationResult.created_at.desc()).first()

        if running_result:
            logger.info(f"Found running simulation for file {file_id}: task {running_result.task_id}")
            return running_result

        # If no running simulation, get the most recent completed simulation
        # Use created_at for ordering since completed_at might be NULL
        result = db.query(SimulationResult).filter(
            SimulationResult.file_id == file_id,
            SimulationResult.user_id == current_user.id,
            SimulationResult.status == "completed"
        ).order_by(SimulationResult.created_at.desc()).first()

        if result:
            logger.info(f"Found completed simulation for file {file_id}: results present = {result.results is not None}")

        return result

    except Exception as e:
        logger.error(f"Error fetching simulation results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")


@app.get("/api/projects/{project_id}/simulations")
async def get_project_simulations(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all completed simulations for a project"""
    try:
        # Use created_at for ordering since completed_at might be NULL
        simulations = db.query(SimulationResult).filter(
            SimulationResult.project_id == project_id,
            SimulationResult.user_id == current_user.id,
            SimulationResult.status == "completed"
        ).order_by(SimulationResult.created_at.desc()).all()
        
        return {
            "simulations": [
                {
                    "id": sim.id,
                    "task_id": sim.task_id,
                    "file_id": sim.file_id,
                    "created_at": sim.created_at.isoformat(),
                    "completed_at": sim.completed_at.isoformat() if sim.completed_at else None,
                    "config": sim.simulation_config,
                    "results": sim.results
                }
                for sim in simulations
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching project simulations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch simulations: {str(e)}")


@app.post("/api/compare-simulations")
async def compare_simulations(
    sim1_id: int,
    sim2_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare two simulations"""
    try:
        # Fetch both simulations
        sim1 = db.query(SimulationResult).filter(
            SimulationResult.id == sim1_id,
            SimulationResult.user_id == current_user.id
        ).first()
        
        sim2 = db.query(SimulationResult).filter(
            SimulationResult.id == sim2_id,
            SimulationResult.user_id == current_user.id
        ).first()
        
        if not sim1 or not sim2:
            raise HTTPException(status_code=404, detail="One or both simulations not found")
        
        return {
            "simulation1": {
                "id": sim1.id,
                "task_id": sim1.task_id,
                "config": sim1.simulation_config,
                "results": sim1.results,
                "completed_at": sim1.completed_at.isoformat() if sim1.completed_at else None
            },
            "simulation2": {
                "id": sim2.id,
                "task_id": sim2.task_id,
                "config": sim2.simulation_config,
                "results": sim2.results,
                "completed_at": sim2.completed_at.isoformat() if sim2.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing simulations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare simulations: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)