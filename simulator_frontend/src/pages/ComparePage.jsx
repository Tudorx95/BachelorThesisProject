import React, { useState, useEffect } from 'react';
import { ArrowLeft, GitCompare, Loader2 } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ComparePage({ onBack, token, activeProjectId }) {
    const [simulations, setSimulations] = useState([]);
    const [sim1, setSim1] = useState(null);
    const [sim2, setSim2] = useState(null);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Auth headers
    const authHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };

    // Load simulations when component mounts
    useEffect(() => {
        if (activeProjectId) {
            loadSimulations();
        }
    }, [activeProjectId]);

    const loadSimulations = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(
                `${API_URL}/api/projects/${activeProjectId}/simulations`,
                { headers: authHeaders }
            );

            if (!res.ok) {
                throw new Error('Failed to load simulations');
            }

            const data = await res.json();
            setSimulations(data.simulations || []);
        } catch (error) {
            console.error('Failed to load simulations:', error);
            setError('Failed to load simulations. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleCompare = async () => {
        if (!sim1 || !sim2) {
            alert('Please select two simulations to compare');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await fetch(
                `${API_URL}/api/compare-simulations?sim1_id=${sim1}&sim2_id=${sim2}`,
                {
                    method: 'POST',
                    headers: authHeaders
                }
            );

            if (!res.ok) {
                throw new Error('Failed to compare simulations');
            }

            const data = await res.json();
            setResults(data);
        } catch (error) {
            console.error('Failed to compare simulations:', error);
            setError('Failed to compare simulations. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-screen bg-gray-50 p-6 overflow-auto">
            {/* Header */}
            <div className="flex items-center gap-4 mb-6">
                <button
                    onClick={onBack}
                    className="p-2 hover:bg-gray-200 rounded transition-colors"
                >
                    <ArrowLeft className="w-5 h-5" />
                </button>
                <h1 className="text-2xl font-bold text-gray-800">Compare Simulations</h1>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
                    {error}
                </div>
            )}

            {/* Loading State */}
            {loading && !results && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                    <span className="ml-3 text-gray-600">Loading simulations...</span>
                </div>
            )}

            {/* Simulation Selectors */}
            {!loading && simulations.length > 0 && (
                <>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-white rounded-lg p-4 shadow border border-gray-200">
                            <label className="block text-sm font-medium mb-2 text-gray-700">
                                Simulation 1:
                            </label>
                            <select
                                value={sim1 || ''}
                                onChange={(e) => setSim1(Number(e.target.value))}
                                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="">-- Choose Simulation --</option>
                                {simulations.map(s => (
                                    <option key={s.id} value={s.id}>
                                        {new Date(s.completed_at).toLocaleString()} - Task: {s.task_id.substring(0, 8)}...
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="bg-white rounded-lg p-4 shadow border border-gray-200">
                            <label className="block text-sm font-medium mb-2 text-gray-700">
                                Simulation 2:
                            </label>
                            <select
                                value={sim2 || ''}
                                onChange={(e) => setSim2(Number(e.target.value))}
                                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="">-- Choose Simulation --</option>
                                {simulations.map(s => (
                                    <option key={s.id} value={s.id}>
                                        {new Date(s.completed_at).toLocaleString()} - Task: {s.task_id.substring(0, 8)}...
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Compare Button */}
                    <button
                        onClick={handleCompare}
                        disabled={!sim1 || !sim2 || loading}
                        className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors shadow"
                    >
                        {loading ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Comparing...
                            </>
                        ) : (
                            <>
                                <GitCompare className="w-5 h-5" />
                                Compare Simulations
                            </>
                        )}
                    </button>
                </>
            )}

            {/* No Simulations Message */}
            {!loading && simulations.length === 0 && (
                <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded">
                    <p className="font-medium">No completed simulations found</p>
                    <p className="text-sm mt-1">Run some simulations first to be able to compare them.</p>
                </div>
            )}

            {/* Results */}
            {results && (
                <div className="grid grid-cols-2 gap-4 mt-6">
                    <ResultBox
                        title={`Simulation 1 (${results.simulation1.task_id.substring(0, 8)}...)`}
                        data={results.simulation1}
                    />
                    <ResultBox
                        title={`Simulation 2 (${results.simulation2.task_id.substring(0, 8)}...)`}
                        data={results.simulation2}
                    />
                </div>
            )}
        </div>
    );
}

function ResultBox({ title, data }) {
    const analysis = data.results?.analysis || {};
    const summary = data.results?.summary || 'No summary available';
    const config = data.config || {};

    return (
        <div className="bg-white rounded-lg p-6 shadow border border-gray-200">
            <h3 className="text-lg font-bold mb-4 text-gray-800 border-b pb-2">{title}</h3>

            {/* Configuration */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">📊 FL Configuration:</h4>
                <div className="space-y-1 text-sm text-gray-600 bg-blue-50 p-3 rounded border border-blue-200">
                    <div className="flex justify-between">
                        <strong>Total Clients (N):</strong>
                        <span className="font-mono text-blue-700">{config.N || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                        <strong>Malicious Clients (M):</strong>
                        <span className="font-mono text-red-600">{config.M || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                        <strong>Neural Network:</strong>
                        <span className="font-mono">{config.NN_NAME || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                        <strong>Training Rounds:</strong>
                        <span className="font-mono text-green-700">{config.ROUNDS || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                        <strong>Poisoned Rounds (R):</strong>
                        <span className="font-mono text-orange-600">{config.R || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                        <strong>Strategy:</strong>
                        <span className="font-mono">{config.strategy || 'N/A'}</span>
                    </div>
                </div>
            </div>

            {/* Data Poisoning Configuration */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">🦠 Data Poisoning Attack:</h4>
                <div className="space-y-2 text-sm bg-red-50 p-3 rounded border border-red-200">
                    {/* Poisoning Operation */}
                    <div className="p-2 bg-white rounded border border-red-200">
                        <div className="flex justify-between items-center">
                            <strong className="text-gray-700">Operation:</strong>
                            <span className="font-mono text-red-700">
                                {config.poison_operation === 'noise' && '🔊 Gaussian Noise'}
                                {config.poison_operation === 'label_flip' && '🔄 Label Flip'}
                                {config.poison_operation === 'backdoor' && '🚪 Backdoor Trigger'}
                                {!config.poison_operation && 'N/A'}
                            </span>
                        </div>
                        {config.poison_operation && (
                            <div className="mt-1 text-xs text-gray-500">
                                <code className="bg-gray-100 px-2 py-0.5 rounded">{config.poison_operation}</code>
                            </div>
                        )}
                    </div>

                    {/* Attack Intensity */}
                    <div className="p-2 bg-white rounded border border-orange-200">
                        <div className="flex justify-between items-center mb-1">
                            <strong className="text-gray-700">Intensity:</strong>
                            <span className="font-mono text-orange-700 text-base font-bold">
                                {config.poison_intensity ? (config.poison_intensity * 100).toFixed(1) : 'N/A'}%
                            </span>
                        </div>
                        {config.poison_intensity !== undefined && (
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div
                                    className="bg-gradient-to-r from-orange-400 to-red-500 h-2 rounded-full transition-all"
                                    style={{ width: `${config.poison_intensity * 100}%` }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Poisoned Data Percentage */}
                    <div className="p-2 bg-white rounded border border-red-300">
                        <div className="flex justify-between items-center mb-1">
                            <strong className="text-gray-700">Data Percentage:</strong>
                            <span className="font-mono text-red-700 text-base font-bold">
                                {config.poison_percentage ? (config.poison_percentage * 100).toFixed(1) : 'N/A'}%
                            </span>
                        </div>
                        {config.poison_percentage !== undefined && (
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div
                                    className="bg-gradient-to-r from-red-500 to-red-700 h-2 rounded-full transition-all"
                                    style={{ width: `${config.poison_percentage * 100}%` }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Attack Summary */}
                    {config.poison_operation && (
                        <div className="p-2 bg-red-100 rounded border border-red-300 text-xs text-red-800">
                            <strong>Summary:</strong> Using <strong>{config.poison_operation}</strong> with{' '}
                            <strong>{config.poison_intensity ? (config.poison_intensity * 100).toFixed(1) : 0}%</strong> intensity on{' '}
                            <strong>{config.poison_percentage ? (config.poison_percentage * 100).toFixed(1) : 0}%</strong> of data{' '}
                            for <strong>{config.R || 0}</strong> rounds with <strong>{config.M || 0}</strong> malicious clients.
                        </div>
                    )}
                </div>
            </div>

            {/* Results */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">📈 Simulation Results:</h4>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between p-2 bg-green-50 rounded border border-green-200">
                        <strong className="text-gray-700">Clean Accuracy:</strong>
                        <span className="font-mono text-green-700 text-base font-bold">
                            {analysis.clean_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className="flex justify-between p-2 bg-red-50 rounded border border-red-200">
                        <strong className="text-gray-700">Poisoned Accuracy:</strong>
                        <span className="font-mono text-red-700 text-base font-bold">
                            {analysis.poisoned_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className="flex justify-between p-2 bg-orange-50 rounded border border-orange-200">
                        <strong className="text-gray-700">Accuracy Drop:</strong>
                        <span className="font-mono text-orange-700 text-base font-bold">
                            {analysis.accuracy_drop?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className="flex justify-between p-2 bg-blue-50 rounded border border-blue-200">
                        <strong className="text-gray-700">GPU Used:</strong>
                        <span className="font-mono text-blue-700">
                            {analysis.gpu_used || 'N/A'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Summary */}
            <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">📝 Summary:</h4>
                <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded overflow-auto max-h-64 font-mono">
                    {summary}
                </pre>
            </div>

            {/* Timestamp */}
            {data.completed_at && (
                <div className="mt-4 pt-3 border-t text-xs text-gray-500">
                    ⏱️ Completed: {new Date(data.completed_at).toLocaleString()}
                </div>
            )}
        </div>
    );
}