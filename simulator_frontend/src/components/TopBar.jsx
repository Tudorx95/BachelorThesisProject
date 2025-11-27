import React from 'react';
import { Menu, Settings, LogOut, User, GitCompare } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function TopBar({
    isSidebarOpen,
    setIsSidebarOpen,
    activeFile,
    onShowSimulationOptions,
    onShowComparePage,
    activeProjectId
}) {
    const { user, logout } = useAuth();

    return (
        <div className="h-16 bg-white border-b border-gray-200 flex items-center px-4">
            {/* Left section */}
            <div className="flex items-center gap-2">
                <button
                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Toggle Sidebar"
                >
                    <Menu className="w-5 h-5 text-gray-600" />
                </button>

                <button
                    onClick={onShowSimulationOptions}
                    className="flex items-center gap-2 px-3 py-2 bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-lg transition-colors font-medium"
                    title="Simulation Options"
                >
                    <Settings className="w-4 h-4" />
                    <span className="text-sm">Simulation Options</span>
                </button>

                <button
                    onClick={onShowComparePage}
                    disabled={!activeProjectId}
                    className="flex items-center gap-2 px-3 py-2 bg-purple-50 text-purple-600 hover:bg-purple-100 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Compare Simulations"
                >
                    <GitCompare className="w-4 h-4" />
                    <span className="text-sm">Compare</span>
                </button>
            </div>

            {/* Center section - Active file name */}
            <div className="flex-1 flex justify-center">
                {activeFile && (
                    <span className="text-sm text-gray-600 font-medium px-4 py-2 bg-gray-50 rounded-lg">
                        {activeFile.name}
                    </span>
                )}
            </div>

            {/* Right section - User info and logout */}
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg">
                    <User className="w-4 h-4 text-gray-600" />
                    <span className="text-sm text-gray-700 font-medium">{user?.username}</span>
                </div>
                <button
                    onClick={logout}
                    className="flex items-center gap-2 px-3 py-2 bg-red-50 text-red-600 hover:bg-red-100 rounded-lg transition-colors"
                    title="Logout"
                >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm font-medium">Logout</span>
                </button>
            </div>
        </div>
    );
}