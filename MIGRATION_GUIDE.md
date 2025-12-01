# EXAID Frontend Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the vanilla HTML/CSS/JS EXAID frontend to the new Next.js 14 (App Router) implementation with TypeScript, Tailwind CSS, and Zustand state management.

## Architecture Changes

### Before (Vanilla JS)
- **State Management**: Global Maps (`agentTraces`, `summaryCards`) with direct DOM manipulation
- **WebSocket**: Manual connection management with vanilla WebSocket API
- **UI Updates**: Direct DOM mutations via `document.getElementById()`, `appendChild()`, etc.
- **Styling**: Global CSS with CSS variables
- **Server**: FastAPI serving static HTML/CSS/JS files

### After (Next.js 14)
- **State Management**: Zustand global store with React state subscriptions
- **WebSocket**: Singleton service with store integration
- **UI Updates**: React state-driven re-renders (no DOM manipulation)
- **Styling**: Tailwind CSS with custom theme extending original color palette
- **Server**: FastAPI backend (unchanged) + separate Next.js dev server

## Prerequisites

- Node.js 18+ installed
- Python environment with FastAPI and EXAID dependencies
- Both servers running simultaneously during development

## Setup Instructions

### 1. Install Next.js Frontend

The Next.js project has been created at `c:\Users\abemk\source\repos\AbemKW\ExAID\demos/frontend\`

Navigate to the frontend directory:

```powershell
cd c:\Users\abemk\source\repos\AbemKW\ExAID\demos/frontend
```

Dependencies are already installed:
- Next.js 14
- TypeScript
- Tailwind CSS
- Zustand (state management)
- react-focus-lock (modal accessibility)
- framer-motion (animations)

### 2. Configure Environment Variables

The `.env.local` file has been created with:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Important**: These must match your FastAPI server URLs. If running on different ports or hosts, update accordingly.

### 3. Update Backend CORS Configuration

CORS middleware has been added to `web_ui/server.py` to allow requests from the Next.js dev server:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For production**: Update `allow_origins` to your production domain.

### 4. Run Both Servers

You need TWO terminal windows running simultaneously:

#### Terminal 1: FastAPI Backend
```powershell
# From project root
cd c:\Users\abemk\source\repos\AbemKW\ExAID
python -m uvicorn demos.backend.server:app --reload
```

Expected output:
```
Starting Reasoning Traces UI server...
Open http://localhost:8000 in your browser
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2: Next.js Frontend
```powershell
# From frontend directory
cd c:\Users\abemk\source\repos\AbemKW\ExAID\demos/frontend
npm run dev
```

Expected output:
```
▲ Next.js 14.x.x
- Local:        http://localhost:3000
- Ready in XXXms
```

### 5. Access the Application

Open your browser to: **http://localhost:3000**

The Next.js frontend will connect to the FastAPI backend at `localhost:8000` via WebSocket and REST API.

## Verification Checklist

Test all functionality to ensure behavior matches the original:

### ✓ WebSocket Connection
- [ ] Connection status shows "Connected" in header (green with pulse animation)
- [ ] Status updates to "Reconnecting (1/5)" if backend restarts
- [ ] After 5 failed attempts, status shows "Disconnected" (red)

### ✓ Case Submission
- [ ] Textarea auto-resizes as you type (max 200px height)
- [ ] Submit button changes to ⏳ emoji when processing
- [ ] Button is disabled during processing
- [ ] Input clears after successful submission

### ✓ Agent Traces
- [ ] New agent window appears when first token arrives from that agent
- [ ] Tokens stream in real-time (50ms batching)
- [ ] Agent window auto-scrolls to bottom as new tokens arrive
- [ ] Auto-scroll STOPS if user manually scrolls up (threshold: 50px from bottom)
- [ ] Auto-scroll RESUMES when user scrolls back to bottom
- [ ] Most recently active agent moves to top of list
- [ ] Clicking header expands/collapses agent window
- [ ] Default height: 200px, expanded height: 400px
- [ ] Expand icon (▼) rotates 180° on expansion
- [ ] "View Full" button opens modal with complete trace

### ✓ Summaries
- [ ] New summary card appears at top when EXAID generates summary
- [ ] New summary auto-expands, all others auto-collapse (accordion behavior)
- [ ] Clicking header toggles expand/collapse
- [ ] Only one summary can be expanded at a time
- [ ] Timestamp formats correctly (HH:MM:SS AM/PM)
- [ ] All 6 fields display correctly:
  - Status / Action
  - Key Findings
  - Differential & Rationale
  - Uncertainty / Confidence
  - Recommendation / Next Step
  - Agent Contributions

### ✓ Modal Behavior
- [ ] Modal opens when clicking "View Full" on agent window
- [ ] Modal displays agent ID and full trace text
- [ ] Clicking backdrop closes modal
- [ ] Pressing ESC key closes modal
- [ ] Clicking X button closes modal
- [ ] Body scroll locked when modal open
- [ ] Focus trapped within modal (Tab key cycles through elements)

### ✓ Animations & Styling
- [ ] Agent windows fade in when created
- [ ] Summary cards fade in when created
- [ ] Expand/collapse animates smoothly (300ms duration)
- [ ] Connection status dot pulses when connected
- [ ] Colors match original design (blue for agents, teal for summaries)
- [ ] Responsive layout works on mobile (single column below 768px)

### ✓ State Management
- [ ] All traces clear when submitting new case
- [ ] All summaries clear when submitting new case
- [ ] Modal closes when submitting new case
- [ ] Count badges update correctly (e.g., "3 traces", "2 summaries")

## Key Behavioral Differences

### What Changed
- **React Strict Mode**: Disabled to prevent double WebSocket connections during development
- **Token Batching**: 50ms batching added to optimize React re-renders (was immediate in vanilla JS)
- **Animations**: Using Framer Motion instead of CSS transitions for smoother height animations

### What Stayed Exactly the Same
- WebSocket reconnection logic (fixed 3s delay, 5 max attempts)
- Auto-scroll threshold detection (50px from bottom)
- Summary accordion behavior (only one expanded)
- Modal escape/backdrop click behavior
- Timestamp formatting
- Agent window repositioning (newest to top)

## Troubleshooting

### WebSocket Connection Issues

**Problem**: Status shows "Disconnected" or constant reconnection attempts

**Solutions**:
1. Verify FastAPI server is running on port 8000
2. Check browser console for CORS errors
3. Ensure `.env.local` has correct WebSocket URL
4. Check firewall isn't blocking WebSocket connections

### CORS Errors

**Problem**: Console shows "Access-Control-Allow-Origin" errors

**Solutions**:
1. Verify CORS middleware is added to `server.py`
2. Confirm `allow_origins` includes `http://localhost:3000`
3. Restart FastAPI server after adding CORS middleware

### Hot Reload Creating Duplicate Connections

**Problem**: Multiple WebSocket connections shown in browser dev tools

**Solutions**:
1. Verify `reactStrictMode: false` in `next.config.ts`
2. This is a dev-only issue; production builds won't have this problem
3. WebSocket service includes safeguards to prevent duplicate connections

### Animations Not Working

**Problem**: Agent windows or summaries don't animate smoothly

**Solutions**:
1. Ensure `framer-motion` is installed: `npm install framer-motion`
2. Check browser console for errors
3. Verify Tailwind config includes animation keyframes

### Token Streaming Lag

**Problem**: Tokens appear in bursts instead of smoothly

**Solutions**:
1. This is expected behavior due to 50ms batching for performance
2. Adjust batch interval in `store/cdssStore.ts` if needed (line: `setInterval(..., 50)`)
3. Lower values (30ms) = smoother but more re-renders
4. Higher values (100ms) = more batching but potentially laggy feel

## File Structure Reference

```
demos/frontend/
├── app/
│   ├── layout.tsx          # Root layout with modal portal
│   ├── page.tsx            # Main page with WebSocket initialization
│   └── globals.css         # Global Tailwind styles
├── components/
│   ├── Header.tsx          # Connection status header
│   ├── CaseInput.tsx       # Textarea form with auto-resize
│   ├── AgentWindow.tsx     # Memoized agent trace window
│   ├── AgentWindowContent.tsx  # Auto-scroll logic
│   ├── AgentTracesPanel.tsx    # Left panel container
│   ├── SummaryCard.tsx     # Memoized summary card
│   ├── SummariesPanel.tsx  # Right panel container
│   └── AgentModal.tsx      # Portal-based modal with focus lock
├── lib/
│   ├── types.ts            # TypeScript interfaces
│   └── websocket.ts        # Singleton WebSocket service
├── store/
│   └── cdssStore.ts        # Zustand global state management
├── .env.local              # Environment variables
├── next.config.ts          # Next.js config (Strict Mode disabled)
└── tailwind.config.ts      # Tailwind theme extending original colors
```

## Performance Optimizations

### Implemented
- **React.memo**: `AgentWindow` and `SummaryCard` components memoized
- **Zustand Selectors**: Components subscribe only to needed state slices
- **Token Batching**: 50ms interval reduces re-render frequency
- **requestAnimationFrame**: Ensures smooth scrolling after DOM updates
- **Framer Motion**: Hardware-accelerated animations

### Optional Future Enhancements
- **Virtual Scrolling**: For traces exceeding 100KB, consider `react-window`
- **Adaptive Batching**: Adjust batch interval based on token frequency
- **Service Worker**: For offline support and caching
- **Error Boundaries**: Wrap components to prevent full app crashes

## Migration Completion

### When to Remove Old Files

**DO NOT** remove vanilla frontend files until you've verified:
1. All checklist items above pass ✓
2. Production deployment tested
3. All stakeholders have approved new UI
4. Old URLs documented for redirect setup

### Files to Remove (After Verification)
```
web_ui/
├── index.html          # Old landing page
├── css/
│   └── style.css       # Old global styles
└── js/
    └── app.js          # Old WebSocket + DOM logic
```

### FastAPI Route Cleanup (Optional)

After migration, you can remove these routes from `server.py`:
```python
@app.get("/")
async def read_root():
    return FileResponse(Path(__file__).parent / "index.html")

app.mount("/css", StaticFiles(...))
app.mount("/js", StaticFiles(...))
```

**Important**: Keep these routes until Next.js frontend is fully deployed to production.

## Production Deployment

### Environment Variables
Update `.env.local` (or production environment):
```env
NEXT_PUBLIC_WS_URL=wss://your-domain.com/ws
NEXT_PUBLIC_API_URL=https://your-domain.com
```

### Backend CORS
Update `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Build Next.js
```powershell
cd demos/frontend
npm run build
npm run start  # Production server on port 3000
```

### Recommended Stack
- **Frontend**: Vercel (automatic Next.js optimization)
- **Backend**: Cloud Run / EC2 / DigitalOcean
- **WebSocket**: Ensure load balancer supports WebSocket upgrades

## Support & Questions

### Common Questions

**Q: Why is React Strict Mode disabled?**
A: Strict Mode causes components to mount twice in development, creating duplicate WebSocket connections. This is purely a development issue.

**Q: Can I use the old frontend and new frontend simultaneously?**
A: Yes! They connect to the same backend independently. Access old frontend at `localhost:8000`, new frontend at `localhost:3000`.

**Q: What happens if I modify the backend?**
A: As long as WebSocket message formats and REST API contracts remain unchanged, the frontend will continue working.

**Q: Can I change the token batch interval?**
A: Yes, edit `store/cdssStore.ts` and change the `setInterval` delay (currently 50ms).

## Success Criteria

Migration is complete when:
- ✅ All verification checklist items pass
- ✅ WebSocket reconnection behaves identically to vanilla version
- ✅ Auto-scroll threshold detection works correctly
- ✅ Summary accordion maintains "only one expanded" behavior
- ✅ Modal accessibility meets standards (focus lock, ESC key, ARIA)
- ✅ Performance is equal or better than vanilla version
- ✅ No console errors in browser dev tools
- ✅ Responsive design works on mobile and desktop

---

**Migration Date**: November 30, 2025  
**Next.js Version**: 14.2.33  
**Backend**: FastAPI (unchanged)  
**State Management**: Zustand 4.x  
**Styling**: Tailwind CSS 3.x

