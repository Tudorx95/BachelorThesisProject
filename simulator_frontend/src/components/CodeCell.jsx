// CodeCell.jsx (updated with Copy button)
import React, { useState } from 'react';
import { Play, Loader2, Copy, Check } from 'lucide-react';

export default function CodeCell({ content, handleContentChange, handleRun, isRunning, isCompleted }) {
    const [copied, setCopied] = useState(false);

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
        <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
            <div className="flex items-center justify-between p-3 bg-gray-50 border-b border-gray-200">
                <span className="text-sm font-medium text-gray-600">Code</span>
                <div className="flex items-center gap-2">
                    {/* Copy Button */}
                    <button
                        onClick={handleCopy}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-gray-700 text-sm rounded hover:bg-gray-200 transition-colors"
                        title="Copy code"
                    >
                        {copied ? (
                            <>
                                <Check className="w-4 h-4 text-green-600" />
                                <span className="text-green-600">Copied!</span>
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
                        className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
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
            <textarea
                value={content}
                onChange={(e) => handleContentChange(e.target.value)}
                className="w-full p-4 font-mono text-sm focus:outline-none resize-none"
                style={{ minHeight: '300px' }}
                placeholder="Write your code here..."
                disabled={isCompleted}
            />
        </div>
    );
}