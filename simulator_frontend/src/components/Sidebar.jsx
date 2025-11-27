import React, { useState, useEffect } from 'react';
import { File, X, ChevronDown, ChevronRight, Plus, Folder, FolderOpen, FileSpreadsheet } from 'lucide-react';

export default function Sidebar({
    isOpen,
    projects,
    activeProjectId,
    activeFileId,
    onSelectProject,
    onSelectFile,
    onDeleteProject,
    onDeleteFile,
    onCreateProject,
    onCreateFile,
    onShowMultiExport
}) {
    const [expandedProjects, setExpandedProjects] = useState(new Set([activeProjectId]));
    const [showNewProjectInput, setShowNewProjectInput] = useState(false);
    const [newProjectName, setNewProjectName] = useState('');
    const [showNewFileInput, setShowNewFileInput] = useState(null);
    const [newFileName, setNewFileName] = useState('');

    // Handle ESC key to cancel project/file creation
    useEffect(() => {
        const handleEsc = (event) => {
            if (event.key === 'Escape') {
                if (showNewProjectInput) {
                    setShowNewProjectInput(false);
                    setNewProjectName('');
                }
                if (showNewFileInput !== null) {
                    setShowNewFileInput(null);
                    setNewFileName('');
                }
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [showNewProjectInput, showNewFileInput]);

    if (!isOpen) return null;

    const toggleProject = (projectId) => {
        const newExpanded = new Set(expandedProjects);
        if (newExpanded.has(projectId)) {
            newExpanded.delete(projectId);
        } else {
            newExpanded.add(projectId);
        }
        setExpandedProjects(newExpanded);
    };

    const handleCreateProject = () => {
        if (newProjectName.trim()) {
            onCreateProject(newProjectName);
            setNewProjectName('');
            setShowNewProjectInput(false);
        }
    };

    const handleCreateFile = (projectId) => {
        if (newFileName.trim()) {
            onCreateFile(projectId, newFileName);
            setNewFileName('');
            setShowNewFileInput(null);
        }
    };

    return (
        <div className="w-64 bg-white border-r border-gray-200 flex flex-col h-full relative z-20">
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-semibold text-gray-700">PROJECTS</h2>
                    <button
                        onClick={() => setShowNewProjectInput(true)}
                        className="p-1 hover:bg-gray-100 rounded transition-colors"
                        title="New Project"
                    >
                        <Plus className="w-4 h-4 text-gray-600" />
                    </button>
                </div>

                {/* Multi-Export Button */}
                <button
                    onClick={onShowMultiExport}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all shadow-sm hover:shadow-md text-sm font-medium mb-3"
                    title="Export Multiple Simulations to CSV"
                >
                    <FileSpreadsheet className="w-4 h-4" />
                    <span>Multi-Export CSV</span>
                </button>

                {/* New Project Input */}
                {showNewProjectInput && (
                    <div className="flex gap-1 mt-2 relative z-30">
                        <input
                            type="text"
                            value={newProjectName}
                            onChange={(e) => setNewProjectName(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
                            placeholder="Project name..."
                            className="flex-1 px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-lg"
                            autoFocus
                        />
                        <button
                            onClick={handleCreateProject}
                            className="px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                        >
                            ✓
                        </button>
                        <button
                            onClick={() => {
                                setShowNewProjectInput(false);
                                setNewProjectName('');
                            }}
                            className="px-2 py-1 bg-gray-200 text-gray-600 text-xs rounded hover:bg-gray-300"
                        >
                            ✕
                        </button>
                    </div>
                )}
            </div>

            {/* Projects List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {projects.length === 0 ? (
                    <div className="text-center py-8 text-gray-500 text-sm">
                        <Folder className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                        <p>No projects yet</p>
                        <p className="text-xs mt-1">Create your first project</p>
                    </div>
                ) : (
                    projects.map(project => {
                        const isExpanded = expandedProjects.has(project.id);
                        const isActive = activeProjectId === project.id;

                        return (
                            <div key={project.id} className="space-y-1">
                                {/* Project Header */}
                                <div
                                    className={`flex items-center justify-between p-2 rounded cursor-pointer group ${isActive ? 'bg-blue-50' : 'hover:bg-gray-50'
                                        }`}
                                >
                                    <div
                                        className="flex items-center gap-2 flex-1"
                                        onClick={() => {
                                            toggleProject(project.id);
                                            onSelectProject(project.id);
                                        }}
                                    >
                                        {isExpanded ? (
                                            <ChevronDown className="w-4 h-4 text-gray-500" />
                                        ) : (
                                            <ChevronRight className="w-4 h-4 text-gray-500" />
                                        )}
                                        {isExpanded ? (
                                            <FolderOpen className="w-4 h-4 text-blue-500" />
                                        ) : (
                                            <Folder className="w-4 h-4 text-blue-500" />
                                        )}
                                        <span className={`text-sm truncate flex-1 ${isActive ? 'text-blue-600 font-medium' : 'text-gray-700'}`}>
                                            {project.name}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setShowNewFileInput(project.id);
                                            }}
                                            className="p-1 hover:bg-blue-100 rounded"
                                            title="New File"
                                        >
                                            <Plus className="w-3 h-3 text-blue-600" />
                                        </button>
                                        {projects.length > 1 && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (window.confirm(`Delete project "${project.name}"?`)) {
                                                        onDeleteProject(project.id);
                                                    }
                                                }}
                                                className="p-1 hover:bg-red-100 rounded"
                                                title="Delete Project"
                                            >
                                                <X className="w-3 h-3 text-red-600" />
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* New File Input */}
                                {showNewFileInput === project.id && (
                                    <div className="ml-6 flex gap-1 mb-2 relative z-30">
                                        <input
                                            type="text"
                                            value={newFileName}
                                            onChange={(e) => setNewFileName(e.target.value)}
                                            onKeyPress={(e) => e.key === 'Enter' && handleCreateFile(project.id)}
                                            placeholder="file.md"
                                            className="flex-1 px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-lg"
                                            autoFocus
                                        />
                                        <button
                                            onClick={() => handleCreateFile(project.id)}
                                            className="px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                                        >
                                            ✓
                                        </button>
                                        <button
                                            onClick={() => {
                                                setShowNewFileInput(null);
                                                setNewFileName('');
                                            }}
                                            className="px-2 py-1 bg-gray-200 text-gray-600 text-xs rounded hover:bg-gray-300"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                )}

                                {/* Files List */}
                                {isExpanded && project.files && (
                                    <div className="ml-6 space-y-1">
                                        {project.files.length === 0 ? (
                                            <p className="text-xs text-gray-400 py-2 px-2">No files yet</p>
                                        ) : (
                                            project.files.map(file => (
                                                <div
                                                    key={file.id}
                                                    className={`flex items-center justify-between p-2 rounded cursor-pointer group ${activeFileId === file.id
                                                        ? 'bg-blue-100 text-blue-700'
                                                        : 'hover:bg-gray-50 text-gray-700'
                                                        }`}
                                                    onClick={() => onSelectFile(file.id, project.id)}
                                                >
                                                    <div className="flex items-center gap-2 flex-1 min-w-0">
                                                        <File className="w-3 h-3 flex-shrink-0" />
                                                        <span className="text-sm truncate">{file.name}</span>
                                                    </div>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            if (window.confirm(`Delete file "${file.name}"?`)) {
                                                                onDeleteFile(file.id);
                                                            }
                                                        }}
                                                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded flex-shrink-0"
                                                        title="Delete File"
                                                    >
                                                        <X className="w-3 h-3 text-red-600" />
                                                    </button>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}