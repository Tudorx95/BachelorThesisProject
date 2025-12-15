// CodeCell.jsx (updated with Monaco Editor for persistent syntax highlighting)
import React, { useState } from 'react';
import { Play, Loader2, Copy, Check } from 'lucide-react';
import Editor, { loader } from '@monaco-editor/react';
import { useTheme } from '../context/ThemeContext';

// Configure Monaco to use local workers from node_modules
loader.config({
    paths: {
        vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs'
    }
});

export default function CodeCell({ content, handleContentChange, handleRun, isRunning, isCompleted }) {
    const [copied, setCopied] = useState(false);
    const { isDarkMode } = useTheme();

    const handleCopy = async () => {
        try {
            // Încearcă metoda modernă (funcționează doar pe HTTPS sau localhost)
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(content);
            } else {
                // Fallback pentru HTTP (funcționează și pe servere remote fără HTTPS)
                const textArea = document.createElement('textarea');
                textArea.value = content;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();

                try {
                    document.execCommand('copy');
                } finally {
                    document.body.removeChild(textArea);
                }
            }

            setCopied(true);
            setTimeout(() => setCopied(false), 2000); // Reset după 2 secunde
        } catch (err) {
            console.error('Failed to copy:', err);
            alert('Nu s-a putut copia textul. Vă rugăm selectați manual și copiați cu Ctrl+C.');
        }
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-300">Code</span>
                <div className="flex items-center gap-2">
                    {/* Copy Button */}
                    <button
                        onClick={handleCopy}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-gray-700 dark:text-gray-300 text-sm rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                        title="Copy code"
                    >
                        {copied ? (
                            <>
                                <Check className="w-4 h-4 text-green-600 dark:text-green-400" />
                                <span className="text-green-600 dark:text-green-400">Copied!</span>
                            </>
                        ) : (
                            <>
                                <Copy className="w-4 h-4" />
                                <span>Copy</span>
                            </>
                        )}
                    </button>

                    {/* Run Button */}
                    <button
                        onClick={handleRun}
                        disabled={isRunning || isCompleted}
                        className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 dark:bg-blue-700 text-white text-sm rounded hover:bg-blue-700 dark:hover:bg-blue-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                    >
                        {isCompleted ? (
                            'Simulation Completed - No Run'
                        ) : isRunning ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Running...
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4" />
                                Run
                            </>
                        )}
                    </button>
                </div>
            </div>
            <div className="flex-1 overflow-hidden">
                <Editor
                    height="100%"
                    defaultLanguage="python"
                    language="python"
                    value={content}
                    onChange={(value) => handleContentChange(value || '')}
                    theme={isDarkMode ? 'vs-dark' : 'light'}
                    options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        tabSize: 4,
                        wordWrap: 'on',
                        readOnly: isCompleted,
                        scrollbar: {
                            vertical: 'auto',
                            horizontal: 'auto',
                        },
                        padding: { top: 10, bottom: 10 },
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                    }}
                    loading={
                        <div className="flex items-center justify-center h-full bg-white dark:bg-gray-800">
                            <Loader2 className="w-6 h-6 animate-spin text-blue-600 dark:text-blue-400" />
                        </div>
                    }
                />
            </div>
        </div>
    );
}