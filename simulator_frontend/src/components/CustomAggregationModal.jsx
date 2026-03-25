import React, { useState, useRef } from 'react';
import { X, Check, AlertTriangle, Loader2 } from 'lucide-react';
import Editor from '@monaco-editor/react';

const TEMPLATE_CODE = `import numpy as np

def custom_aggregate(client_weights, client_sizes, global_weights, num_malicious, **kwargs):
    """
    Custom aggregation function for Federated Learning.

    Args:
        client_weights: list of weight arrays from each client
                        Each element is a list of numpy arrays (one per layer)
        client_sizes:   list of dataset sizes per client
        global_weights: current global model weights (list of numpy arrays)
        num_malicious:  number of expected malicious clients

    Returns:
        aggregated_weights: list of numpy arrays (same structure as client_weights[0])
    """
    # Example: FedAvg (weighted average based on dataset size)
    total_size = sum(client_sizes)
    avg_weights = [np.zeros_like(w, dtype=np.float64) for w in client_weights[0]]

    for cw, size in zip(client_weights, client_sizes):
        weight = size / total_size
        for i, w in enumerate(cw):
            avg_weights[i] += w * weight

    return [avg_weights[i].astype(w.dtype) for i, w in enumerate(client_weights[0])]
`;

export default function CustomAggregationModal({ onClose, onSave, apiUrl, token }) {
    const [functionName, setFunctionName] = useState('');
    const [code, setCode] = useState(TEMPLATE_CODE);
    const [error, setError] = useState(null);
    const [isValidating, setIsValidating] = useState(false);
    const [success, setSuccess] = useState(false);
    const editorRef = useRef(null);

    const handleEditorDidMount = (editor) => {
        editorRef.current = editor;
    };

    const sanitizeName = (name) => {
        // Allow only alphanumeric + underscore, convert spaces to underscores
        return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '').toLowerCase();
    };

    const handleValidateAndUpload = async () => {
        setError(null);
        setSuccess(false);

        // Validate function name
        const sanitized = sanitizeName(functionName);
        if (!sanitized || sanitized.length < 2) {
            setError('Please enter a valid function name (at least 2 characters, alphanumeric + underscores).');
            return;
        }

        // Basic client-side check: code must contain "def custom_aggregate"
        if (!code.includes('def custom_aggregate')) {
            setError('The code must contain a function named "custom_aggregate".');
            return;
        }

        setIsValidating(true);

        try {
            const response = await fetch(`${apiUrl}/api/upload-aggregation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    function_name: sanitized,
                    code: code
                })
            });

            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                setError(data.detail || data.message || 'Validation failed. Please check your code.');
                setIsValidating(false);
                return;
            }

            setSuccess(true);
            setIsValidating(false);

            // After a short delay, close and return the function name
            setTimeout(() => {
                onSave({ name: sanitized, code: code });
            }, 800);

        } catch (err) {
            setError(`Connection error: ${err.message}`);
            setIsValidating(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-[60] p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-gray-200 dark:border-gray-700">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20">
                    <div>
                        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">
                            ⚙️ Define Custom Aggregation Function
                        </h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            Write your own aggregation method in Python. It will be validated and uploaded to the server.
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 bg-red-100 hover:bg-red-200 dark:bg-red-900/40 dark:hover:bg-red-900/60 rounded-lg transition-colors"
                        title="Cancel"
                    >
                        <X className="w-5 h-5 text-red-600 dark:text-red-400" />
                    </button>
                </div>

                {/* Function name input */}
                <div className="px-6 py-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Function Name
                    </label>
                    <input
                        type="text"
                        value={functionName}
                        onChange={(e) => setFunctionName(e.target.value)}
                        placeholder="e.g., my_trimmed_krum"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400"
                    />
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                        This name will appear as <span className="font-mono text-purple-600 dark:text-purple-400">@{sanitizeName(functionName) || 'function_name'}</span> in the aggregation methods list.
                    </p>
                </div>

                {/* Monaco Editor */}
                <div className="flex-1 min-h-0">
                    <Editor
                        height="400px"
                        defaultLanguage="python"
                        value={code}
                        onChange={(value) => setCode(value || '')}
                        onMount={handleEditorDidMount}
                        theme="vs-dark"
                        options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: 'on',
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            tabSize: 4,
                            wordWrap: 'on',
                            padding: { top: 12 }
                        }}
                    />
                </div>

                {/* Status messages */}
                {error && (
                    <div className="mx-6 mt-3 p-3 bg-red-50 dark:bg-red-900/30 rounded-lg border border-red-200 dark:border-red-800 flex items-start gap-2">
                        <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap">{error}</p>
                    </div>
                )}

                {success && (
                    <div className="mx-6 mt-3 p-3 bg-green-50 dark:bg-green-900/30 rounded-lg border border-green-200 dark:border-green-800">
                        <p className="text-sm text-green-700 dark:text-green-300 font-medium">
                            ✅ Function validated and uploaded successfully!
                        </p>
                    </div>
                )}

                {/* Action buttons */}
                <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700">
                    <button
                        onClick={onClose}
                        className="px-5 py-2.5 bg-red-100 hover:bg-red-200 dark:bg-red-900/40 dark:hover:bg-red-900/60 text-red-700 dark:text-red-300 rounded-lg font-medium transition-colors flex items-center gap-2"
                    >
                        <X className="w-4 h-4" />
                        Cancel
                    </button>

                    <button
                        onClick={handleValidateAndUpload}
                        disabled={isValidating || success}
                        className={`px-5 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                            isValidating || success
                                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                                : 'bg-green-600 hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-600 text-white'
                        }`}
                    >
                        {isValidating ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Validating...
                            </>
                        ) : success ? (
                            <>
                                <Check className="w-4 h-4" />
                                Uploaded!
                            </>
                        ) : (
                            <>
                                <Check className="w-4 h-4" />
                                Validate & Upload
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
