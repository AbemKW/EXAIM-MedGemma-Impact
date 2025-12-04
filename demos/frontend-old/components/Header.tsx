'use client';

import { useWsStatus, useCDSSStore } from '@/store/cdssStore';

export default function Header() {
  const wsStatus = useWsStatus();
  const reconnectAttempts = useCDSSStore((state) => state.reconnectAttempts);

  const getStatusClass = () => {
    switch (wsStatus) {
      case 'connected':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'connecting':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'disconnected':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusText = () => {
    if (wsStatus === 'connecting' && reconnectAttempts > 0) {
      return `Reconnecting (${reconnectAttempts}/5)`;
    }
    return wsStatus.charAt(0).toUpperCase() + wsStatus.slice(1);
  };

  const getStatusDotClass = () => {
    switch (wsStatus) {
      case 'connected':
        return 'bg-green-500 animate-pulse';
      case 'connecting':
        return 'bg-yellow-500';
      case 'disconnected':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <header className="flex justify-between items-center py-5 border-b border-gray-200">
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold text-blue-600 tracking-tight">EXAID</h1>
        <p className="text-sm text-gray-600 font-medium">
          Clinical Decision Support System
        </p>
      </div>
      <div className="flex items-center">
        <div
          className={`px-4 py-2 rounded-full text-xs font-semibold uppercase tracking-wide flex items-center gap-2 border ${getStatusClass()}`}
        >
          <span className={`w-2 h-2 rounded-full ${getStatusDotClass()}`} />
          {getStatusText()}
        </div>
      </div>
    </header>
  );
}
