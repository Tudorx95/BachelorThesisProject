import React, { useState, useEffect, useRef } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { SimulationProvider, useSimulation } from './context/SimulationContext';
import Login from './pages/Login';
import Register from './pages/Register';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import CodeCell from './components/CodeCell';
import OutputCell from './components/OutputCell';
import SimulationOptions from './components/SimulationOptions';
import ComparePage from './pages/ComparePage';
import MultiExportCSV from './components/MultiExportCSV';

function AppContent() {
    const { user, token, loading: authLoading, isAuthenticated } = useAuth();

    // Use SimulationContext pentru persistență
    const {
        config: simulationConfig,
        setConfig: setSimulationConfig,
        activeSimulation,
        simulationOutput,
        startSimulation,
        updateSimulationStep,
        updateProgress,
        completeSimulation,
        failSimulation,
        stopSimulation,
        clearSimulationOutput,
        fileSimulationStates,
        setFileSimulationStates,
        completedSimulations,
        setCompletedSimulations,
        activeProjectId,
        setActiveProjectId,
        activeFileId,
        setActiveFileId
    } = useSimulation();

    const [showLogin, setShowLogin] = useState(true);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    // Projects and files state
    const [projects, setProjects] = useState([]);

    const [showSimulationOptions, setShowSimulationOptions] = useState(false);
    const [pendingRun, setPendingRun] = useState(false);
    const [showComparePage, setShowComparePage] = useState(false);
    const [showMultiExport, setShowMultiExport] = useState(false);

    // Store WebSocket connections per taskId for multiple simultaneous simulations
    const wsRefs = useRef({});
    const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

    // Auth headers
    const authHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };

    // Get active file
    const activeProject = projects.find(p => p.id === activeProjectId);
    const activeFile = activeProject?.files?.find(f => f.id === activeFileId);

    // Get simulation state for active file
    const activeFileSimState = fileSimulationStates[activeFileId] || {
        isRunning: false,
        isCancelling: false,
        currentTaskId: null,
        orchestratorStatus: null
    };

    // Helper function to update file simulation state
    const updateFileSimState = (fileId, updates) => {
        setFileSimulationStates(prev => ({
            ...prev,
            [fileId]: {
                ...(prev[fileId] || {}),
                ...updates
            }
        }));
    };

    // Cleanup WebSocket on unmount
    useEffect(() => {
        return () => {
            // Close all active WebSocket connections
            Object.values(wsRefs.current).forEach(ws => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            });
            wsRefs.current = {};
        };
    }, []);

    // Load projects when authenticated
    useEffect(() => {
        if (isAuthenticated) {
            loadProjects();
        }
    }, [isAuthenticated]);

    // Load simulation results when active file changes
    useEffect(() => {
        if (activeFileId && isAuthenticated) {
            loadSimulationResults(activeFileId);
        }
    }, [activeFileId, isAuthenticated]);

    // Run simulation after config is saved if pending
    useEffect(() => {
        if (simulationConfig && pendingRun) {
            setPendingRun(false);
            handleRun();
        }
    }, [simulationConfig]);

    // Handle ESC to close SimulationOptions
    useEffect(() => {
        const handleEsc = (event) => {
            if (event.key === 'Escape' && showSimulationOptions) {
                setShowSimulationOptions(false);
                setPendingRun(false);
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [showSimulationOptions]);

    // Load projects from backend
    const loadProjects = async () => {
        try {
            const response = await fetch(`${API_URL}/api/projects`, {
                headers: authHeaders
            });

            if (!response.ok) throw new Error('Failed to load projects');

            const projectsData = await response.json();

            // Load files for each project
            const projectsWithFiles = await Promise.all(
                projectsData.map(async (project) => {
                    const filesResponse = await fetch(`${API_URL}/api/projects/${project.id}/files`, {
                        headers: authHeaders
                    });
                    const files = await filesResponse.json();
                    return { ...project, files };
                })
            );

            setProjects(projectsWithFiles);

            // Set active project and file if not set
            if (!activeProjectId && projectsWithFiles.length > 0) {
                setActiveProjectId(projectsWithFiles[0].id);
                if (projectsWithFiles[0].files.length > 0) {
                    setActiveFileId(projectsWithFiles[0].files[0].id);
                }
            }
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    };

    // Check if task is still running on the backend
    const checkTaskStatus = async (taskId) => {
        try {
            const response = await fetch(`${API_URL}/task-status/${taskId}`, {
                headers: authHeaders
            });

            if (!response.ok) {
                if (response.status === 404) {
                    console.log(`[Reconnect] Task ${taskId} not found on server`);
                    return null;
                }
                throw new Error('Failed to check task status');
            }

            const status = await response.json();
            console.log(`[Reconnect] Task ${taskId} status:`, status);
            return status;
        } catch (error) {
            console.error('[Reconnect] Error checking task status:', error);
            return null;
        }
    };

    // Load simulation results for active file
    const loadSimulationResults = async (fileId) => {
        try {
            const response = await fetch(`${API_URL}/api/files/${fileId}/simulation-results`, {
                headers: authHeaders
            });

            if (!response.ok) {
                // No results found is ok, just return
                if (response.status === 404) return;
                throw new Error('Failed to load simulation results');
            }

            const result = await response.json();
            console.log(`[Load] Received result from backend for file ${fileId}:`, result);

            if (result && (result.results || result.status === 'running')) {
                // Restore the output if it exists
                if (result.output) {
                    const updatedProjects = projects.map(p => ({
                        ...p,
                        files: p.files.map(f =>
                            f.id === fileId ? { ...f, output: result.output } : f
                        )
                    }));
                    setProjects(updatedProjects);
                }

                // Check if this is a completed simulation or still running
                if (result.status === 'completed') {
                    console.log(`[Load] Restoring completed simulation with results:`, result.results);

                    // Restore the orchestrator status from saved results
                    updateFileSimState(fileId, {
                        orchestratorStatus: {
                            status: 'completed',
                            results_data: result.results,
                            step: 7,
                            message: 'Simulation completed (restored from database)'
                        }
                    });

                    // Mark as completed
                    setCompletedSimulations(prev => ({
                        ...prev,
                        [fileId]: true
                    }));

                    console.log(`[Load] Loaded completed simulation results for file ${fileId}`);
                } else if (result.task_id && result.status === 'running') {
                    // Task is still running, check backend status and reconnect WebSocket
                    console.log(`[Load] Found running task ${result.task_id} for file ${fileId}`);

                    const taskStatus = await checkTaskStatus(result.task_id);

                    if (taskStatus && taskStatus.status === 'running') {
                        // Task is still running on backend, reconnect WebSocket
                        console.log(`[Reconnect] Reconnecting WebSocket for task ${result.task_id}`);

                        updateFileSimState(fileId, {
                            isRunning: true,
                            currentTaskId: result.task_id,
                            orchestratorStatus: taskStatus.orchestrator_status || {
                                status: 'running',
                                step: taskStatus.current_step || 1,
                                message: 'Reconnected to running simulation'
                            }
                        });

                        // Reconnect WebSocket
                        connectWebSocket(result.task_id, fileId, '');
                    } else {
                        // Task is no longer running, mark as completed or error
                        console.log(`[Reconnect] Task ${result.task_id} is no longer running`);
                        updateFileSimState(fileId, {
                            isRunning: false,
                            currentTaskId: null,
                            orchestratorStatus: {
                                status: taskStatus?.status || 'completed',
                                step: 7,
                                message: 'Simulation finished while offline',
                                results_data: taskStatus?.results || result.results
                            }
                        });

                        if (taskStatus?.status === 'completed') {
                            setCompletedSimulations(prev => ({
                                ...prev,
                                [fileId]: true
                            }));
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error loading simulation results:', error);
        }
    };

    // Create new project
    const handleCreateProject = async (name) => {
        try {
            const response = await fetch(`${API_URL}/api/projects`, {
                method: 'POST',
                headers: authHeaders,
                body: JSON.stringify({ name, description: '' })
            });

            if (!response.ok) throw new Error('Failed to create project');

            const newProject = await response.json();
            newProject.files = [];
            setProjects([...projects, newProject]);
            setActiveProjectId(newProject.id);
        } catch (error) {
            console.error('Error creating project:', error);
            alert('Failed to create project');
        }
    };

    // Delete project
    const handleDeleteProject = async (projectId) => {
        try {
            const response = await fetch(`${API_URL}/api/projects/${projectId}`, {
                method: 'DELETE',
                headers: authHeaders
            });

            if (!response.ok) throw new Error('Failed to delete project');

            const newProjects = projects.filter(p => p.id !== projectId);
            setProjects(newProjects);

            if (activeProjectId === projectId && newProjects.length > 0) {
                setActiveProjectId(newProjects[0].id);
                if (newProjects[0].files.length > 0) {
                    setActiveFileId(newProjects[0].files[0].id);
                }
            }
        } catch (error) {
            console.error('Error deleting project:', error);
            alert('Failed to delete project');
        }
    };

    // Create new file
    const handleCreateFile = async (projectId, name) => {
        try {
            const response = await fetch(`${API_URL}/api/projects/${projectId}/files`, {
                method: 'POST',
                headers: authHeaders,
                body: JSON.stringify({
                    name,
                    content: '# New File\n\n## Write your code here\n'
                })
            });

            if (!response.ok) throw new Error('Failed to create file');

            const newFile = await response.json();

            const updatedProjects = projects.map(p => {
                if (p.id === projectId) {
                    return { ...p, files: [...p.files, newFile] };
                }
                return p;
            });

            setProjects(updatedProjects);
            setActiveProjectId(projectId);
            setActiveFileId(newFile.id);
        } catch (error) {
            console.error('Error creating file:', error);
            alert('Failed to create file');
        }
    };

    // Delete file
    const handleDeleteFile = async (fileId) => {
        try {
            const response = await fetch(`${API_URL}/api/files/${fileId}`, {
                method: 'DELETE',
                headers: authHeaders
            });

            if (!response.ok) throw new Error('Failed to delete file');

            const updatedProjects = projects.map(p => ({
                ...p,
                files: p.files.filter(f => f.id !== fileId)
            }));

            setProjects(updatedProjects);

            // Select another file if current was deleted
            if (activeFileId === fileId) {
                const currentProject = updatedProjects.find(p => p.id === activeProjectId);
                if (currentProject && currentProject.files.length > 0) {
                    setActiveFileId(currentProject.files[0].id);
                } else {
                    setActiveFileId(null);
                }
            }

            // Clean up simulation state for deleted file
            setFileSimulationStates(prev => {
                const newState = { ...prev };
                delete newState[fileId];
                return newState;
            });

            // Close WebSocket for this file's task if it exists
            const fileSimState = fileSimulationStates[fileId];
            if (fileSimState && fileSimState.currentTaskId) {
                const ws = wsRefs.current[fileSimState.currentTaskId];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
                delete wsRefs.current[fileSimState.currentTaskId];
            }
        } catch (error) {
            console.error('Error deleting file:', error);
            alert('Failed to delete file');
        }
    };

    // Select project
    const handleSelectProject = (projectId) => {
        setActiveProjectId(projectId);
        const project = projects.find(p => p.id === projectId);
        if (project && project.files.length > 0) {
            setActiveFileId(project.files[0].id);
        } else {
            setActiveFileId(null);
        }
    };

    // Select file
    const handleSelectFile = (fileId) => {
        setActiveFileId(fileId);
    };

    // Handle content change
    const handleContentChange = async (newContent) => {
        if (!activeFile) return;

        try {
            const response = await fetch(`${API_URL}/api/files/${activeFile.id}`, {
                method: 'PUT',
                headers: authHeaders,
                body: JSON.stringify({ content: newContent })
            });

            if (!response.ok) throw new Error('Failed to update file');

            const updatedProjects = projects.map(p => ({
                ...p,
                files: p.files.map(f =>
                    f.id === activeFile.id ? { ...f, content: newContent } : f
                )
            }));

            setProjects(updatedProjects);
        } catch (error) {
            console.error('Error updating file:', error);
        }
    };

    // Save simulation results to database
    const saveSimulationResults = async (fileId, taskId, config, results, status = 'completed', output = null) => {
        try {
            const project = projects.find(p => p.files.some(f => f.id === fileId));
            if (!project) {
                console.error('Project not found for file', fileId);
                return;
            }

            // Get current output from the file if not provided
            if (!output) {
                const file = project.files.find(f => f.id === fileId);
                output = file?.output || null;
            }

            const response = await fetch(`${API_URL}/api/simulation-results`, {
                method: 'POST',
                headers: authHeaders,
                body: JSON.stringify({
                    file_id: fileId,
                    project_id: project.id,
                    task_id: taskId,
                    simulation_config: config,
                    results: results,
                    output: output,
                    status: status
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save simulation results');
            }

            const savedResult = await response.json();
            console.log('Simulation results saved successfully:', savedResult);
        } catch (error) {
            console.error('Error saving simulation results:', error);
        }
    };

    // Connect to WebSocket for orchestrator updates
    const connectWebSocket = (taskId, fileId, initialOutput) => {
        const ws = new WebSocket(`${WS_URL}/ws/${taskId}`);

        // Store this WebSocket with its taskId
        wsRefs.current[taskId] = ws;

        console.log(`[WS] Creating WebSocket for task ${taskId}, file ${fileId}`);

        ws.onopen = () => {
            console.log(`[WS] Connected for task ${taskId}`);
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log(`[WS] Raw message for task ${taskId}:`, message);

            // Backend sends { type: "orchestrator_update", data: {...} } or { type: "connected", ... }
            if (message.type === 'connected') {
                console.log(`[WS] Connection confirmed for task ${taskId}`);
                return;
            }

            if (message.type === 'orchestrator_update' && message.data) {
                const data = message.data;
                console.log(`[WS] Orchestrator update for task ${taskId}:`, data);

                // Update orchestrator status for the specific file
                // Note: data might have results_data when completed
                updateFileSimState(fileId, {
                    orchestratorStatus: data
                });

                // If simulation completed or encountered an error
                if (data.status === 'completed' || data.status === 'error' || data.status === 'cancelled') {
                    console.log(`[WS] Simulation ${data.status} for task ${taskId}`);

                    updateFileSimState(fileId, {
                        isRunning: false,
                        currentTaskId: null
                    });

                    if (data.status === 'completed') {
                        setCompletedSimulations(prev => ({
                            ...prev,
                            [fileId]: true
                        }));

                        // Save simulation results to database
                        if (data.results_data) {
                            saveSimulationResults(
                                fileId,
                                taskId,
                                simulationConfig,
                                data.results_data,
                                'completed'
                            );
                        }
                    }

                    // Close and cleanup WebSocket
                    ws.close();
                    delete wsRefs.current[taskId];
                }
            }
        };

        ws.onerror = (error) => {
            console.error(`[WS] Error for task ${taskId}:`, error);
            updateFileSimState(fileId, {
                isRunning: false,
                currentTaskId: null
            });
            delete wsRefs.current[taskId];
        };

        ws.onclose = () => {
            console.log(`[WS] Closed for task ${taskId}`);
            // Cleanup reference
            delete wsRefs.current[taskId];
        };
    };

    // Cancel simulation
    const handleCancelSimulation = async () => {
        const taskId = activeFileSimState.currentTaskId;

        // Helper function pentru cleanup local
        const performLocalCleanup = (message) => {
            console.log(`[Cancel] Performing local cleanup: ${message}`);

            // Update orchestrator status to show cancellation
            updateFileSimState(activeFileId, {
                orchestratorStatus: {
                    status: 'cancelled',
                    message: message,
                    step: activeFileSimState.orchestratorStatus?.step || 1
                }
            });

            // Close and cleanup WebSocket connection
            if (taskId) {
                const ws = wsRefs.current[taskId];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
                delete wsRefs.current[taskId];
            }

            // Update file output
            const updatedProjects = projects.map(p => ({
                ...p,
                files: p.files.map(f =>
                    f.id === activeFileId ? {
                        ...f,
                        output: (f.output || '') + '\n\n=== Cancelled ===\n' + message
                    } : f
                )
            }));
            setProjects(updatedProjects);

            // Reset states
            updateFileSimState(activeFileId, {
                isRunning: false,
                isCancelling: false,
                currentTaskId: null
            });
        };

        // Dacă nu există task ID, forțează cleanup local
        if (!taskId) {
            console.warn('[Cancel] No task ID available - performing force cleanup');
            performLocalCleanup('Task cancelled locally (no active task ID)');
            return;
        }

        console.log(`[Cancel] Attempting to cancel task ${taskId}`);

        updateFileSimState(activeFileId, {
            isCancelling: true
        });

        try {
            const response = await fetch(`${API_URL}/cancel/${taskId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            // Dacă taskul nu există pe backend (404), face cleanup local
            if (response.status === 404) {
                console.warn('[Cancel] Task not found on backend - performing local cleanup');
                performLocalCleanup('Task not found on server (possibly already completed or cancelled)');
                return;
            }

            const data = await response.json();

            if (data.status === 'success') {
                console.log('[Cancel] Simulation cancelled successfully:', data);
                performLocalCleanup(data.message || 'Simulation cancelled by user');
            } else {
                console.error('[Cancel] Cancellation failed:', data);
                // Chiar dacă backend-ul spune că a eșuat, încearcă cleanup local
                performLocalCleanup(`Cancellation response: ${data.detail || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('[Cancel] Error cancelling simulation:', error);

            // Dacă este eroare de network sau task nu există, face cleanup local oricum
            if (error.message.includes('fetch') || error.message.includes('NetworkError')) {
                console.warn('[Cancel] Network error - performing local cleanup anyway');
                performLocalCleanup('Task cancelled locally (server unreachable)');
            } else {
                // Pentru alte erori, arată mesaj dar tot face cleanup
                alert(`Error cancelling simulation: ${error.message}\nPerforming local cleanup anyway.`);
                performLocalCleanup('Task cancelled with errors');
            }
        }
    };

    // Run simulation
    const handleRun = async () => {
        if (!activeFile) return;

        if (completedSimulations[activeFileId]) {
            alert('Simulation already completed for this file. No more runs allowed.');
            return;
        }

        // Check if simulation config exists
        if (!simulationConfig) {
            setShowSimulationOptions(true);
            setPendingRun(true);
            return;
        }

        console.log(`[Run] Starting simulation for file ${activeFileId}`);

        updateFileSimState(activeFileId, {
            isRunning: true,
            isCancelling: false,
            currentTaskId: null,
            orchestratorStatus: null
        });

        try {
            console.log(`[Run] Sending POST request to ${API_URL}/run`);
            console.log(`[Run] Simulation config:`, simulationConfig);

            const response = await fetch(`${API_URL}/run`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    filename: activeFile.name,
                    code: activeFile.content,
                    simulation_config: simulationConfig
                }),
            });

            console.log(`[Run] Response status: ${response.status}`);
            const data = await response.json();
            console.log(`[Run] Data received:`, data);

            if (data.status === 'error') {
                const updatedProjects = projects.map(p => ({
                    ...p,
                    files: p.files.map(f =>
                        f.id === activeFileId ? { ...f, output: data.output } : f
                    )
                }));
                setProjects(updatedProjects);
                updateFileSimState(activeFileId, {
                    isRunning: false
                });
                return;
            }

            // Update output with initial execution result
            const updatedProjects = projects.map(p => ({
                ...p,
                files: p.files.map(f =>
                    f.id === activeFileId ? { ...f, output: data.output } : f
                )
            }));
            setProjects(updatedProjects);

            // Connect WebSocket to receive orchestrator updates
            if (data.task_id) {
                console.log(`[Run] Got task_id: ${data.task_id}, connecting WebSocket`);
                updateFileSimState(activeFileId, {
                    currentTaskId: data.task_id
                });

                // Save initial task info to database for reconnection after refresh
                await saveSimulationResults(
                    activeFileId,
                    data.task_id,
                    simulationConfig,
                    null, // No results yet
                    'running'
                );

                // Pass fileId to connectWebSocket
                connectWebSocket(data.task_id, activeFileId, data.output);
            } else {
                console.log('[Run] No task_id received, simulation finished');
                updateFileSimState(activeFileId, {
                    isRunning: false
                });
            }

        } catch (error) {
            console.error('[Run] Error:', error);
            const updatedProjects = projects.map(p => ({
                ...p,
                files: p.files.map(f =>
                    f.id === activeFileId ? { ...f, output: `Error: ${error.message}` } : f
                )
            }));
            setProjects(updatedProjects);
            updateFileSimState(activeFileId, {
                isRunning: false
            });
        }
    };

    // Show loading screen
    if (authLoading) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-50">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading...</p>
                </div>
            </div>
        );
    }

    // Show auth screens if not authenticated
    if (!isAuthenticated) {
        return showLogin ? (
            <Login onSwitchToRegister={() => setShowLogin(false)} />
        ) : (
            <Register onSwitchToLogin={() => setShowLogin(true)} />
        );
    }

    // Main application
    // Show ComparePage if requested
    if (showComparePage) {
        return (
            <ComparePage
                onBack={() => setShowComparePage(false)}
                token={token}
                activeProjectId={activeProjectId}
            />
        );
    }

    return (
        <div className="flex h-screen bg-gray-50">
            <Sidebar
                isOpen={isSidebarOpen}
                projects={projects}
                activeProjectId={activeProjectId}
                activeFileId={activeFileId}
                onSelectProject={handleSelectProject}
                onSelectFile={handleSelectFile}
                onDeleteProject={handleDeleteProject}
                onDeleteFile={handleDeleteFile}
                onCreateProject={handleCreateProject}
                onCreateFile={handleCreateFile}
                onShowMultiExport={() => setShowMultiExport(true)}
            />
            <div className="flex-1 flex flex-col">
                <TopBar
                    isSidebarOpen={isSidebarOpen}
                    setIsSidebarOpen={setIsSidebarOpen}
                    activeFile={activeFile}
                    onShowSimulationOptions={() => setShowSimulationOptions(true)}
                    onShowComparePage={() => setShowComparePage(true)}
                    activeProjectId={activeProjectId}
                />
                <div className="flex-1 overflow-auto p-6">
                    <div className="max-w-5xl mx-auto space-y-4">
                        {activeFile ? (
                            <>
                                <CodeCell
                                    content={activeFile.content || ''}
                                    handleContentChange={handleContentChange}
                                    handleRun={handleRun}
                                    isRunning={activeFileSimState.isRunning}
                                    isCompleted={completedSimulations[activeFileId]}
                                />
                                {/* Afișăm OutputCell doar dacă există output SAU simulare activă pentru ACEST fișier */}
                                {(activeFile.output || activeFileSimState.isRunning || activeFileSimState.orchestratorStatus) && (
                                    <OutputCell
                                        output={activeFile.output || ''}
                                        isLoading={activeFileSimState.isRunning}
                                        orchestratorStatus={activeFileSimState.orchestratorStatus}
                                        onCancel={handleCancelSimulation}
                                        isCancelling={activeFileSimState.isCancelling}
                                        fileName={activeFile.name}
                                    />
                                )}
                            </>
                        ) : (
                            <div className="text-center py-20">
                                <p className="text-gray-500 text-lg">No file selected</p>
                                <p className="text-gray-400 text-sm mt-2">
                                    Create a new project and file to get started
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Simulation Options Modal */}
            {showSimulationOptions && (
                <SimulationOptions
                    onClose={() => {
                        setShowSimulationOptions(false);
                        setPendingRun(false);
                    }}
                    onSave={(config) => {
                        setSimulationConfig(config);
                        console.log('Simulation config saved:', config);
                        setShowSimulationOptions(false);
                    }}
                    initialConfig={simulationConfig}
                />
            )}

            {/* Multi-Export CSV Modal */}
            {showMultiExport && (
                <MultiExportCSV
                    onClose={() => setShowMultiExport(false)}
                    token={token}
                    projects={projects}
                />
            )}
        </div>
    );
}

export default function App() {
    return (
        <AuthProvider>
            <SimulationProvider>
                <AppContent />
            </SimulationProvider>
        </AuthProvider>
    );
}