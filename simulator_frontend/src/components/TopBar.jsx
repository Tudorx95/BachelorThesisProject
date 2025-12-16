import React from 'react';
import { Menu, Settings, LogOut, User, GitCompare, Moon, Sun, LineChart } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

export default function TopBar({
    isSidebarOpen,
    setIsSidebarOpen,
    activeFile,
    onShowSimulationOptions,
    onShowComparePage,
    onShowGraphsPage,
    activeProjectId,
    disableNavigation = false
}) {
    const { user, logout } = useAuth();
    const { isDarkMode, toggleTheme } = useTheme();

    return (
        <div className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center px-4 transition-colors">
            {/* Left section */}
            <div className="flex items-center gap-2">
                <button
                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    disabled={disableNavigation}
                    className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent dark:disabled:hover:bg-transparent"
                    title="Toggle Sidebar"
                >
                    <Menu className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                </button>

                <button
                    onClick={onShowSimulationOptions}
                    disabled={disableNavigation}
                    className="flex items-center gap-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-50 dark:disabled:hover:bg-blue-900/30"
                    title="Simulation Options"
                >
                    <Settings className="w-4 h-4" />
                    <span className="text-sm">Simulation Options</span>
                </button>

                <button
                    onClick={onShowComparePage}
                    disabled={!activeProjectId || disableNavigation}
                    className="flex items-center gap-2 px-3 py-2 bg-purple-50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/50 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-purple-50 dark:disabled:hover:bg-purple-900/30"
                    title="Compare Simulations"
                >
                    <GitCompare className="w-4 h-4" />
                    <span className="text-sm">Compare</span>
                </button>

                <button
                    onClick={onShowGraphsPage}
                    disabled={!activeProjectId || disableNavigation}
                    className="flex items-center gap-2 px-3 py-2 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 hover:bg-green-100 dark:hover:bg-green-900/50 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-green-50 dark:disabled:hover:bg-green-900/30"
                    title="View Simulation Graphs"
                >
                    <LineChart className="w-4 h-4" />
                    <span className="text-sm">Graphs</span>
                </button>
            </div>

            {/* Center section - Active file name */}
            <div className="flex-1 flex justify-center">
                {activeFile && (
                    <span className="text-sm text-gray-600 dark:text-gray-300 font-medium px-4 py-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        {activeFile.name}
                    </span>
                )}
            </div>

            {/* Right section - Theme toggle, User info and logout */}
            <div className="flex items-center gap-3">
                <button
                    onClick={toggleTheme}
                    className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                >
                    {isDarkMode ? (
                        <Sun className="w-5 h-5 text-yellow-400" />
                    ) : (
                        <Moon className="w-5 h-5 text-gray-600" />
                    )}
                </button>

                <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                    <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">{user?.username}</span>
                </div>
                <button
                    onClick={logout}
                    className="flex items-center gap-2 px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/50 rounded-lg transition-colors"
                    title="Logout"
                >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm font-medium">Logout</span>
                </button>
            </div>
        </div>
    );
}