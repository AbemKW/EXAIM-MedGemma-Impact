'use client';

import { useWsStatus, useCDSSStore } from '@/store/cdssStore';
import { Badge } from '@/components/ui/badge';

export default function Header() {
  const wsStatus = useWsStatus();
  const reconnectAttempts = useCDSSStore((state) => state.reconnectAttempts);

  const getStatusVariant = () => {
    switch (wsStatus) {
      case 'connected':
        return 'default';
      case 'connecting':
        return 'secondary';
      case 'disconnected':
        return 'destructive';
      default:
        return 'outline';
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
        return 'bg-green-600/70 animate-pulse';
      case 'connecting':
        return 'bg-amber-600/70';
      case 'disconnected':
        return 'bg-red-700/70';
      default:
        return 'bg-zinc-500';
    }
  };

  return (
    <header className="flex justify-between items-center py-4 border-b border-border flex-shrink-0">
      <div className="flex items-center gap-4">
        <h1 className="text-5xl font-bold text-primary tracking-tight" style={{ fontFamily: 'var(--font-inter), sans-serif' }}>EXAID</h1>
        <p className="text-base text-muted-foreground font-medium" style={{ fontFamily: 'var(--font-inter), sans-serif' }}>
          Clinical Decision Support System
        </p>
      </div>
      <div className="flex items-center">
        <Badge
          variant={getStatusVariant()}
          className="flex items-center gap-2 uppercase tracking-wide text-sm px-3 py-1"
        >
          <span className={`w-2 h-2 rounded-full ${getStatusDotClass()}`} />
          {getStatusText()}
        </Badge>
      </div>
    </header>
  );
}

