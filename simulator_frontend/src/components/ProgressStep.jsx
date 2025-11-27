import React from 'react';
import { Loader2, AlertCircle, CheckCircle, Clock } from 'lucide-react';

export default function ProgressStep({
    step,
    stepName,
    currentStep,
    status,
    message,
    timestamp
}) {
    // Determine the status of this specific step
    const isActive = step === currentStep && status === 'running';
    const isCompleted = step < currentStep || status === 'completed';
    const isError = status === 'error';
    const isPending = step > currentStep && status !== 'completed';

    // Format timestamp if available
    const formatTimestamp = (ts) => {
        if (!ts) return null;
        const date = new Date(ts);
        return date.toLocaleTimeString('ro-RO', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <div className={`
            flex items-start gap-3 p-3 rounded-lg transition-all duration-300
            ${isActive ? 'bg-blue-50 border border-blue-200 shadow-sm' :
                isCompleted ? 'bg-green-50 border border-green-100' :
                    isError ? 'bg-red-50 border border-red-200' :
                        'bg-gray-50 border border-gray-100'}
        `}>
            {/* Step Icon/Number Circle */}
            <div className={`
                flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all
                ${isError ? 'border-red-500 bg-red-100' :
                    isCompleted ? 'border-green-500 bg-green-100' :
                        isActive ? 'border-blue-500 bg-blue-100 shadow-md' :
                            'border-gray-300 bg-gray-100'}
            `}>
                {isError ? (
                    <AlertCircle className="w-6 h-6 text-red-600" />
                ) : isCompleted ? (
                    <CheckCircle className="w-6 h-6 text-green-600" />
                ) : isActive ? (
                    <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                ) : (
                    <span className="text-gray-500 text-sm font-bold">{step}</span>
                )}
            </div>

            {/* Step Content */}
            <div className="flex-1 min-w-0">
                {/* Step Name */}
                <div className="flex items-center justify-between mb-1">
                    <h4 className={`
                        text-sm font-semibold transition-colors
                        ${isError ? 'text-red-700' :
                            isCompleted ? 'text-green-700' :
                                isActive ? 'text-blue-700' :
                                    'text-gray-500'}
                    `}>
                        {stepName}
                        {isCompleted && ' ✓'}
                        {isActive && ' •••'}
                    </h4>

                    {/* Timestamp for active or completed steps */}
                    {(isActive || isCompleted) && timestamp && (
                        <span className="flex items-center gap-1 text-xs text-gray-500">
                            <Clock className="w-3 h-3" />
                            {formatTimestamp(timestamp)}
                        </span>
                    )}
                </div>

                {/* Real-time Message from Orchestrator */}
                {message && (isActive || isError) && (
                    <div className={`
                        text-xs mt-1 animate-fade-in
                        ${isError ? 'text-red-600 font-medium' :
                            isActive ? 'text-blue-600' :
                                'text-gray-600'}
                    `}>
                        {message}
                    </div>
                )}

                {/* Pending status message */}
                {isPending && (
                    <div className="text-xs text-gray-400 italic mt-1">
                        Waiting to start...
                    </div>
                )}

                {/* Progress bar for active step */}
                {isActive && (
                    <div className="mt-2">
                        <div className="h-1 bg-blue-200 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 animate-pulse"
                                style={{ width: '100%' }} />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}