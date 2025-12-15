import React, { useState } from 'react';
import { Settings, X } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function SimulationOptions({ onClose, onSave, initialConfig }) {
    const { isDarkMode } = useTheme();
    const [config, setConfig] = useState(initialConfig || {
        N: 10,
        M: 2,
        NN_NAME: 'SimpleNN',
        R: 5,
        ROUNDS: 10,
        strategy: 'first',
        poison_operation: 'noise',
        poison_intensity: 0.1,
        poison_percentage: 0.2
    });

    const handleChange = (field, value) => {
        setConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSave(config);
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
                <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Settings className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Simulation Options</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-6">
                    {/* FL Simulation Options */}
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-4">Federated Learning Configuration</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Number of Clients (N)
                                </label>
                                <input
                                    type="number"
                                    value={config.N}
                                    onChange={(e) => handleChange('N', parseInt(e.target.value))}
                                    min="1"
                                    max="100"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Total number of participating clients</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Malicious Clients (M)
                                </label>
                                <input
                                    type="number"
                                    value={config.M}
                                    onChange={(e) => handleChange('M', parseInt(e.target.value))}
                                    min="0"
                                    max={config.N}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Number of malicious clients</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Training Rounds (ROUNDS)
                                </label>
                                <input
                                    type="number"
                                    value={config.ROUNDS}
                                    onChange={(e) => handleChange('ROUNDS', parseInt(e.target.value))}
                                    min="1"
                                    max="1000"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Total FL training rounds</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Neural Network Name
                                </label>
                                <input
                                    type="text"
                                    value={config.NN_NAME}
                                    onChange={(e) => handleChange('NN_NAME', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    placeholder="e.g., SimpleNN, ResNet50"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Model architecture name</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Distribution Strategy
                                </label>
                                <select
                                    value={config.strategy}
                                    onChange={(e) => handleChange('strategy', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="first">First - Malicious at start</option>
                                    <option value="last">Last - Malicious at end</option>
                                    <option value="alternate">Alternate - Interleaved</option>
                                    <option value="alternate_data">Alternate Data - Switch datasets</option>
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Client selection strategy</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Poisoned Data Rounds (R)
                                </label>
                                <input
                                    type="number"
                                    value={config.R}
                                    onChange={(e) => handleChange('R', parseInt(e.target.value))}
                                    min="0"
                                    max={config.ROUNDS}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Rounds using poisoned data</p>
                            </div>
                        </div>
                    </div>

                    {/* Data Poisoning Parameters */}
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                        <h3 className="text-lg font-semibold text-red-900 dark:text-red-300 mb-4">Data Poisoning Attack Parameters</h3>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Poisoning Operation
                                </label>
                                <select
                                    value={config.poison_operation}
                                    onChange={(e) => handleChange('poison_operation', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="noise">Gaussian Noise</option>
                                    <option value="label_flip">Label Flip</option>
                                    <option value="backdoor">Backdoor Trigger</option>
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Type of poisoning attack</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Intensity
                                </label>
                                <input
                                    type="number"
                                    value={config.poison_intensity}
                                    onChange={(e) => handleChange('poison_intensity', parseFloat(e.target.value))}
                                    min="0.01"
                                    max="1.0"
                                    step="0.01"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Attack intensity (0.01-1.0)</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Percentage
                                </label>
                                <input
                                    type="number"
                                    value={config.poison_percentage}
                                    onChange={(e) => handleChange('poison_percentage', parseFloat(e.target.value))}
                                    min="0.01"
                                    max="1.0"
                                    step="0.01"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">% of data to poison (0.01-1.0)</p>
                            </div>
                        </div>
                        <div className="mt-3 p-3 bg-red-100 dark:bg-red-900/30 rounded text-sm text-red-800 dark:text-red-300">
                            <strong>Note:</strong> Data poisoning is automatically applied to simulate attacks.
                            Configure the attack parameters above.
                        </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="flex-1 px-4 py-3 bg-blue-600 dark:bg-blue-700 text-white rounded-lg font-semibold hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors"
                        >
                            Save & Apply Configuration
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}