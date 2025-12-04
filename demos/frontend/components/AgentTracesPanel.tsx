'use client';

import { useAllAgents } from '@/store/cdssStore';
import AgentWindow from './AgentWindow';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default function AgentTracesPanel() {
  // Subscribe to agents array directly
  const agents = useAllAgents();

  return (
    <Card className="flex flex-col overflow-hidden h-full">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-blue-950/40 to-blue-950/30 border-b border-blue-900/20 py-3 px-5">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold">Raw Agent Traces</CardTitle>
          <Badge variant="secondary" className="text-sm">
            {agents.length} trace{agents.length !== 1 ? 's' : ''}
          </Badge>
        </div>
      </CardHeader>

      {/* Panel Content */}
      <CardContent className="flex-1 overflow-y-auto p-4 space-y-3">
        {agents.length === 0 ? (
          <p className="text-muted-foreground text-center py-8 text-sm">
            No traces yet. Process a case to see verbose reasoning traces.
          </p>
        ) : (
          agents.map((agent) => (
            <AgentWindow key={agent.id} cardId={agent.id} />
          ))
        )}
      </CardContent>
    </Card>
  );
}

