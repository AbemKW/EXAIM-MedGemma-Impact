'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore, useAgentTrace } from '@/store/cdssStore';
import AgentWindowContent from './AgentWindowContent';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface AgentWindowProps {
  cardId: string;
}

function AgentWindow({ cardId }: AgentWindowProps) {
  const agent = useAgentTrace(cardId);
  const toggleAgent = useCDSSStore((state) => state.toggleAgent);
  const openModal = useCDSSStore((state) => state.openModal);

  if (!agent) return null;

  const handleToggle = () => {
    toggleAgent(cardId);
  };

  const handleViewFull = (e: React.MouseEvent) => {
    e.stopPropagation();
    openModal(cardId, agent.fullText);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="overflow-hidden">
        {/* Header */}
        <CardHeader
          onClick={handleToggle}
          className="cursor-pointer hover:bg-muted/50 transition-colors border-b border-border"
        >
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <CardTitle className="text-base">{agent.agentName}</CardTitle>
              <Badge variant="default" className="bg-emerald-600 hover:bg-emerald-600">
                Active
              </Badge>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={handleViewFull}
                variant="secondary"
                size="sm"
              >
                View Full
              </Button>
              <motion.span
                animate={{ rotate: agent.isExpanded ? 180 : 0 }}
                transition={{ duration: 0.3 }}
                className="text-muted-foreground"
              >
                ▼
              </motion.span>
            </div>
          </div>
        </CardHeader>

        {/* Content */}
        <motion.div
          initial={false}
          animate={{
            height: agent.isExpanded ? '400px' : '200px',
          }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="overflow-hidden"
        >
          <CardContent className="p-0 h-full">
            <AgentWindowContent
              fullText={agent.fullText}
              isExpanded={agent.isExpanded}
            />
          </CardContent>
        </motion.div>
      </Card>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(AgentWindow);

